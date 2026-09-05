#include "LoadObj.h"
#include "ValidateTetMesh.h"
#include "mesh/Tets.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <mutex>
#include <print>
#include <pthread.h>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct CaseSignature {
    std::string Key;
    uint32_t Tets{}, Steiner{}, Flips{}, Splits{}, MissingEdges{}, MissingFaces{};
    uint64_t Hash{};
};

struct ObjectResult {
    std::string Out;
    std::vector<CaseSignature> Signatures;
    int Failures{};
};

struct WorkItem {
    std::string Corpus;
    fs::path Obj;
};

uint64_t MeshHash(const TetMesh &mesh) {
    uint64_t hash = 0xcbf29ce484222325ull;
    const auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= 0x100000001b3ull;
    };
    for (const auto &p : mesh.Points) {
        mix(std::bit_cast<uint64_t>(p.x));
        mix(std::bit_cast<uint64_t>(p.y));
        mix(std::bit_cast<uint64_t>(p.z));
    }
    for (const auto &tet : mesh.Tets) {
        for (const uint32_t vertex : tet) mix(vertex);
    }
    return hash;
}

std::string FormatSignature(const CaseSignature &signature) {
    return std::format(
        "{:<34} {:>8} {:>6} {:>9} {:>6} {:>6} {:>6} {:016x}",
        signature.Key, signature.Tets, signature.Steiner, signature.Flips, signature.Splits,
        signature.MissingEdges, signature.MissingFaces, signature.Hash
    );
}

ObjectResult RunObject(const WorkItem &work) {
    ObjectResult result;
    const auto surface = LoadObj(work.Obj);
    if (!surface || surface->Positions.empty()) {
        result.Out = std::format("{}/{}: failed to load\n", work.Corpus, work.Obj.filename().string());
        result.Failures = 1;
        return result;
    }

    const std::vector<dvec3> input_points(surface->Positions.begin(), surface->Positions.end());
    for (const bool quality : {false, true}) {
        const auto tets = GenerateTets(surface->Positions, surface->TriangleIndices, {.Refinement = quality ? fastfem::TetRefinement::Quality : fastfem::TetRefinement::None});
        if (!tets) {
            result.Out += std::format("{}/{} {}: tetrahedralization failed: {}\n", work.Corpus, work.Obj.stem().string(), quality ? "q" : "noq", tets.error());
            ++result.Failures;
            continue;
        }

        const auto validation = ValidateTetMesh(input_points, surface->TriangleIndices, tets->Mesh);
        if (!validation.empty()) {
            result.Out += std::format("{}/{} {}: invalid mesh: {}\n", work.Corpus, work.Obj.stem().string(), quality ? "q" : "noq", validation);
            ++result.Failures;
        }

        const auto &profile = tets->Profile;
        result.Signatures.push_back({
            .Key = std::format("{}/{}@1.00/{}", work.Corpus, work.Obj.stem().string(), quality ? "q" : "noq"),
            .Tets = profile.TetCount,
            .Steiner = profile.SteinerCount,
            .Flips = profile.FlipCount,
            .Splits = profile.SplitCount,
            .MissingEdges = profile.MissingEdgeCount,
            .MissingFaces = profile.MissingFaceCount,
            .Hash = MeshHash(tets->Mesh),
        });
    }

    result.Out += std::format("{}/{}: {}\n", work.Corpus, work.Obj.stem().string(), result.Failures == 0 ? "OK" : "FAILED");
    return result;
}

std::vector<fs::path> CorpusObjects(const fs::path &directory, const std::vector<std::string> &names) {
    std::vector<fs::path> objects;
    for (const auto &entry : fs::directory_iterator{directory}) {
        if (entry.path().extension() != ".obj") continue;
        const auto stem = entry.path().stem().string();
        if (!names.empty() && std::ranges::find(names, stem) == names.end()) continue;
        objects.push_back(entry.path());
    }
    std::ranges::sort(objects, {}, [](const fs::path &path) {
        const auto stem = path.stem().string();
        return std::pair{std::atoi(stem.c_str()), stem};
    });
    return objects;
}

std::vector<ObjectResult> RunObjects(const std::vector<WorkItem> &work_items, unsigned thread_count) {
    std::vector<ObjectResult> results(work_items.size());
    std::vector<unsigned char> done(work_items.size(), 0);
    std::vector<size_t> order(work_items.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::ranges::sort(order, std::ranges::greater{}, [&](size_t i) { return fs::file_size(work_items[i].Obj); });

    std::atomic<size_t> next{0};
    std::mutex mutex;
    std::condition_variable ready;
    const auto work = [&] {
        for (size_t k = next++; k < work_items.size(); k = next++) {
            const size_t i = order[k];
            results[i] = RunObject(work_items[i]);
            {
                std::lock_guard lock{mutex};
                done[i] = 1;
            }
            ready.notify_all();
        }
    };

    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, 512 * 1024 * 1024);
    std::vector<pthread_t> threads(thread_count);
    size_t started = 0;
    for (; started < threads.size(); ++started) {
        if (pthread_create(
                &threads[started], &attributes,
                [](void *argument) -> void * {
                    (*static_cast<const decltype(work) *>(argument))();
                    return nullptr;
                },
                const_cast<void *>(static_cast<const void *>(&work))
            ) != 0)
            break;
    }
    pthread_attr_destroy(&attributes);
    threads.resize(started);
    if (threads.empty()) work();

    for (size_t i = 0; i < work_items.size(); ++i) {
        std::unique_lock lock{mutex};
        ready.wait(lock, [&, i] { return done[i] != 0; });
        lock.unlock();
        std::print("{}", results[i].Out);
    }
    for (auto &thread : threads) pthread_join(thread, nullptr);
    return results;
}

int CheckSnapshot(
    const fs::path &snapshot_path,
    const std::vector<CaseSignature> &actual,
    const std::set<std::string> &selected_corpora
) {
    std::ifstream file{snapshot_path};
    if (!file) {
        std::println("snapshot is missing: {}", snapshot_path.string());
        return 1;
    }

    std::map<std::string, std::string> expected;
    for (std::string line; std::getline(file, line);) {
        if (line.empty() || line.front() == '#') continue;
        expected.emplace(line.substr(0, line.find(' ')), line);
    }

    int failures = 0;
    std::set<std::string> observed;
    for (const auto &signature : actual) {
        observed.insert(signature.Key);
        const auto it = expected.find(signature.Key);
        if (it == expected.end()) {
            std::println("snapshot: {} is not recorded", signature.Key);
            ++failures;
            continue;
        }
        const auto line = FormatSignature(signature);
        if (line != it->second) {
            std::println("snapshot: {} changed\n    was {}\n    now {}", signature.Key, it->second, line);
            ++failures;
        }
    }

    for (const auto &[key, _] : expected) {
        const auto corpus = key.substr(0, key.find('/'));
        if (selected_corpora.contains(corpus) && !observed.contains(key)) {
            std::println("snapshot: {} did not run", key);
            ++failures;
        }
    }
    std::println("snapshot: {} cases, {} differences", actual.size(), failures);
    return failures;
}

void PrintUsage(const char *program) {
    std::println("Usage: {} [--corpus-dir PATH] [--snapshot PATH] [--dataset NAME] [--jobs N] [object ...]", program);
}
} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    fs::path corpus_root{FASTFEM_TET_CORPUS_DIR};
    fs::path snapshot_path{FASTFEM_TET_SNAPSHOT_PATH};
    std::vector<std::string> datasets, names;
    unsigned jobs = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument{argv[i]};
        if (argument == "--corpus-dir" && i + 1 < argc) corpus_root = argv[++i];
        else if (argument == "--snapshot" && i + 1 < argc) snapshot_path = argv[++i];
        else if (argument == "--dataset" && i + 1 < argc) datasets.emplace_back(argv[++i]);
        else if (argument == "--jobs" && i + 1 < argc) jobs = std::max(1u, unsigned(std::stoul(argv[++i])));
        else if (argument == "--help") {
            PrintUsage(argv[0]);
            return 0;
        } else names.emplace_back(argument);
    }
    if (!fs::is_directory(corpus_root)) {
        std::println("tet corpus root is absent: {} (run script/SetupTetCorpus)", corpus_root.string());
        return 77;
    }
    if (datasets.empty())
        for (const std::string_view dataset : {"realimpact", "thingi10k"})
            if (fs::is_directory(corpus_root / dataset)) datasets.emplace_back(dataset);
    if (datasets.empty()) {
        std::println("tet corpus has no installed datasets: {} (run script/SetupTetCorpus)", corpus_root.string());
        return 77;
    }

    std::vector<WorkItem> work_items;
    std::set<std::string> selected_corpora;
    for (const auto &dataset : datasets) {
        const auto directory = corpus_root / dataset;
        if (!fs::is_directory(directory)) {
            std::println("tet corpus dataset is absent: {}", directory.string());
            return 1;
        }
        selected_corpora.insert(dataset);
        for (auto &object : CorpusObjects(directory, names)) work_items.push_back({dataset, std::move(object)});
    }
    if (work_items.empty()) {
        std::println("no matching OBJ files in {}", corpus_root.string());
        return 1;
    }

    std::println("tet corpus: {} objects, two quality arms, {} workers", work_items.size(), std::min<unsigned>(jobs, work_items.size()));
    auto results = RunObjects(work_items, std::min<unsigned>(jobs, work_items.size()));
    std::vector<CaseSignature> signatures;
    int failures = 0;
    for (auto &result : results) {
        failures += result.Failures;
        signatures.insert(signatures.end(), std::make_move_iterator(result.Signatures.begin()), std::make_move_iterator(result.Signatures.end()));
    }
    std::ranges::sort(signatures, {}, &CaseSignature::Key);
    failures += CheckSnapshot(snapshot_path, signatures, selected_corpora);
    std::println("tet corpus: {} objects, {} failures", results.size(), failures);
    return failures == 0 ? 0 : 1;
}
