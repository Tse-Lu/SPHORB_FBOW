/*
 * train_sporb_fbow.cpp
 *
 * Extract SPHORB panoramic features from images and train a FBOW bag-of-words
 * vocabulary (dictionary). Multiple image directories can be supplied -- all
 * images found recursively (including in subdirectories) are used for training.
 *
 * Usage:
 *   train_sporb_fbow [options] -s <sphorb_path> -o <output.fbow> <dir1> [dir2 ...]
 *
 * Options:
 *   -s, --sphorb <path>     Path to SPHORB folder (must contain Data/ subdirectory)
 *   -o, --output <path>     Output vocabulary file path
 *   -k <int>                Branching factor (default: 10)
 *   -l <int>                Depth levels (default: 6)
 *   -t <int>                Number of threads (default: 1)
 *   -n <int>                Max features per image (default: 500)
 *   --max-iters <int>       Max k-means iterations, 0 = random centers only (default: 0)
 *   -v, --verbose           Verbose output
 *   -h, --help              Show this help
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <set>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include "SPHORB.h"
#include "fbow.h"
#include "vocabulary_creator.h"

// Recognised image file extensions (lowercase).
static bool isImageExt(const std::string &ext) {
    return ext == "jpg"  || ext == "jpeg" || ext == "png"  ||
           ext == "bmp"  || ext == "ppm"  || ext == "pgm"  ||
           ext == "tiff" || ext == "tif"  || ext == "webp";
}

// Recursively collect all image files under `dir`.
static void collectImages(const std::string &dir, std::vector<std::string> &files) {
    DIR *dp = opendir(dir.c_str());
    if (!dp) {
        std::cerr << "Warning: cannot open directory " << dir << std::endl;
        return;
    }
    struct dirent *entry;
    while ((entry = readdir(dp))) {
        std::string name(entry->d_name);
        if (name == "." || name == "..") continue;

        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            collectImages(full, files);
        } else if (S_ISREG(st.st_mode)) {
            std::string ext;
            auto dot = name.find_last_of('.');
            if (dot != std::string::npos) {
                ext = name.substr(dot + 1);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            }
            if (isImageExt(ext)) {
                files.push_back(full);
            }
        }
    }
    closedir(dp);
}

int main(int argc, char **argv) {
    if (argc < 2) {
usage:
        std::cerr << "train_sporb_fbow -- Train a FBOW vocabulary using SPHORB panoramic features\n\n";
        std::cerr << "Usage: " << argv[0] << " [options] -s <sphorb_path> -o <output.fbow> <dir1> [dir2 ...]\n\n";
        std::cerr << "Required:\n";
        std::cerr << "  -s, --sphorb <path>     Path to SPHORB folder (must contain Data/)\n";
        std::cerr << "  -o, --output <path>     Output vocabulary file (.fbow)\n";
        std::cerr << "  <dir1> [dir2 ...]       One or more image directories (recursive)\n\n";
        std::cerr << "Options:\n";
        std::cerr << "  -k <int>                Branching factor (default: 10)\n";
        std::cerr << "  -l <int>                Depth levels (default: 6)\n";
        std::cerr << "  -t <int>                Number of threads (default: 1)\n";
        std::cerr << "  -n <int>                Max features per image (default: 1000)\n";
        std::cerr << "  --max-iters <int>       Max k-means iterations, 0 = random centers (default: 5)\n";
        std::cerr << "  -v, --verbose           Verbose output\n";
        std::cerr << "  -h, --help              Show this help\n\n";
        std::cerr << "Examples:\n";
        std::cerr << "  " << argv[0] << " -s ./SPHORB/ -o vocab.fbow ./pano_images/\n";
        std::cerr << "  " << argv[0] << " -s ./SPHORB/ -o vocab.fbow -k 10 -l 5 -t 8 -n 1000 --max-iters 5 ./city/ ./campus/\n";
        return 1;
    }

    std::vector<std::string> imageDirs;
    std::string sphorbPath;
    std::string outputPath;
    int k = 10, L = 6, nthreads = 1, maxIters = 5, nfeatures = 1000;
    bool verbose = false;

    int i = 1;
    while (i < argc) {
        std::string arg(argv[i]);

        if (arg == "-h" || arg == "--help") {
            goto usage;
        } else if (arg == "-s" || arg == "--sphorb") {
            if (i + 1 < argc) sphorbPath = argv[++i];
            else { std::cerr << "Error: -s requires a path argument." << std::endl; return 1; }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) outputPath = argv[++i];
            else { std::cerr << "Error: -o requires a path argument." << std::endl; return 1; }
        } else if (arg == "-k") {
            if (i + 1 < argc) k = std::stoi(argv[++i]);
            else { std::cerr << "Error: -k requires an integer." << std::endl; return 1; }
        } else if (arg == "-l") {
            if (i + 1 < argc) L = std::stoi(argv[++i]);
            else { std::cerr << "Error: -l requires an integer." << std::endl; return 1; }
        } else if (arg == "-t") {
            if (i + 1 < argc) nthreads = std::stoi(argv[++i]);
            else { std::cerr << "Error: -t requires an integer." << std::endl; return 1; }
        } else if (arg == "-n") {
            if (i + 1 < argc) nfeatures = std::stoi(argv[++i]);
            else { std::cerr << "Error: -n requires an integer." << std::endl; return 1; }
        } else if (arg == "--max-iters") {
            if (i + 1 < argc) maxIters = std::stoi(argv[++i]);
            else { std::cerr << "Error: --max-iters requires an integer." << std::endl; return 1; }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg[0] == '-') {
            std::cerr << "Error: unknown option '" << arg << "'." << std::endl;
            return 1;
        } else {
            // Positional argument -- image directory.
            imageDirs.push_back(arg);
        }
        ++i;
    }

    if (sphorbPath.empty()) {
        std::cerr << "Error: SPHORB path is required (-s <path>)." << std::endl;
        return 1;
    }
    if (outputPath.empty()) {
        std::cerr << "Error: output path is required (-o <path>)." << std::endl;
        return 1;
    }
    if (imageDirs.empty()) {
        std::cerr << "Error: at least one image directory is required." << std::endl;
        return 1;
    }

    // Verify SPHORB Data directory exists.
    {
        std::string dataDir = sphorbPath + "/Data";
        struct stat st;
        if (stat(dataDir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            std::cerr << "Error: SPHORB Data directory not found at " << dataDir << std::endl;
            return 1;
        }
    }

    // Verify image directories and collect images.
    std::set<std::string> imageSet;  // deduplicate by full path
    for (const auto &dir : imageDirs) {
        struct stat st;
        if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            std::cerr << "Error: " << dir << " is not a valid directory." << std::endl;
            return 1;
        }
        std::vector<std::string> dirFiles;
        collectImages(dir, dirFiles);
        std::cout << "  " << dir << ": " << dirFiles.size() << " image(s)" << std::endl;
        for (const auto &f : dirFiles) imageSet.insert(f);
    }

    std::vector<std::string> images(imageSet.begin(), imageSet.end());
    std::sort(images.begin(), images.end());

    if (images.empty()) {
        std::cerr << "Error: no images found in the given directories." << std::endl;
        return 1;
    }
    std::cout << "Total images: " << images.size() << std::endl;

    // Create SPHORB detector.
    std::cout << "Initializing SPHORB detector (nfeatures=" << nfeatures << ")..." << std::endl;
    cv::SPHORB sphorb(sphorbPath, nfeatures, 7, 20);
    std::cout << "  Descriptor size: " << sphorb.descriptorSize() << " bytes ("
              << sphorb.descriptorSize() * 8 << " bits)" << std::endl;

    // Extract features from all images.
    std::vector<cv::Mat> allFeatures;
    size_t totalKeypoints = 0;
    size_t skipped = 0;

    for (size_t idx = 0; idx < images.size(); ++idx) {
        std::cout << "[" << (idx + 1) << "/" << images.size() << "] "
                  << images[idx] << std::endl;

        cv::Mat img = cv::imread(images[idx], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "  WARNING: could not read image, skipping." << std::endl;
            ++skipped;
            continue;
        }

        // SPHORB works on equirectangular panoramic images. Typical sizes are
        // 2000x1000 or similar (width = 2 * height). Warn if aspect ratio
        // deviates significantly.
        double aspect = static_cast<double>(img.cols) / img.rows;
        if (aspect < 1.5 || aspect > 2.5) {
            std::cerr << "  WARNING: image aspect ratio is " << aspect
                      << " (panoramas usually ~2:1). Results may be suboptimal." << std::endl;
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // Use operator() directly instead of detectAndCompute() because
        // OpenCV 4 checks Feature2D::empty() which SPHORB does not override.
        sphorb(img, cv::noArray(), keypoints, descriptors);

        if (!descriptors.empty()) {
            allFeatures.push_back(descriptors.clone());
            totalKeypoints += keypoints.size();
        }
        std::cout << "  " << keypoints.size() << " keypoints extracted" << std::endl;
    }

    if (allFeatures.empty()) {
        std::cerr << "Error: no features extracted from any image." << std::endl;
        return 1;
    }

    std::cout << "\nSummary:\n";
    std::cout << "  Images processed:  " << (images.size() - skipped) << "/" << images.size() << std::endl;
    std::cout << "  Total keypoints:   " << totalKeypoints << std::endl;
    std::cout << "  Avg keypoints/img: " << (totalKeypoints / (images.size() - skipped)) << std::endl;

    // Verify descriptor consistency across images.
    {
        int expectedCols = allFeatures[0].cols;
        int expectedType = allFeatures[0].type();
        for (size_t idx = 1; idx < allFeatures.size(); ++idx) {
            if (allFeatures[idx].cols != expectedCols || allFeatures[idx].type() != expectedType) {
                std::cerr << "Error: inconsistent descriptor dimensions across images." << std::endl;
                return 1;
            }
        }
    }

    // Build vocabulary.
    fbow::VocabularyCreator::Params params;
    params.k        = k;
    params.L        = L;
    params.nthreads = nthreads;
    params.maxIters = maxIters;
    params.verbose  = verbose;

    srand(0);

    long long expectedWords = 1;
    for (int j = 0; j < L; ++j) expectedWords *= k;

    std::cout << "\nBuilding vocabulary (k=" << k << ", L=" << L
              << ", expected words=" << expectedWords
              << ", threads=" << nthreads
              << ", maxIters=" << maxIters << ")..." << std::endl;

    fbow::VocabularyCreator creator;
    fbow::Vocabulary voc;

    auto t0 = std::chrono::high_resolution_clock::now();
    creator.create(voc, allFeatures, "sporb", params);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Training done in " << ms / 1000.0 << " s" << std::endl;
    std::cout << "Vocabulary blocks: " << voc.size() << std::endl;

    // Save vocabulary to file.
    voc.saveToFile(outputPath);
    std::cout << "Vocabulary saved to " << outputPath << std::endl;

    return 0;
}
