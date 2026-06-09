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
 *   -n <int>                Max features per image (default: 1000)
 *   --max-iters <int>       Max k-means iterations, 0 = random centers (default: 5)
 *   -a <int>                Augmented rotations per image (default: 0)
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
#include <cmath>
#include <random>
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

// Rotate an equirectangular panoramic image.
// roll/pitch/yaw in radians.  R = Rz(yaw) * Ry(pitch) * Rx(roll).
static cv::Mat rotateEquirectangular(const cv::Mat &src,
                                     double roll_rad, double pitch_rad, double yaw_rad)
{
    int w = src.cols, h = src.rows;
    cv::Mat map_x(h, w, CV_32F), map_y(h, w, CV_32F);

    double cr = cos(roll_rad),  sr = sin(roll_rad);
    double cp = cos(pitch_rad), sp = sin(pitch_rad);
    double cy = cos(yaw_rad),   sy = sin(yaw_rad);

    // Inverse rotation matrix  Rt = Rx^T * Ry^T * Rz^T
    double rt00 =  cy * cp;
    double rt01 = -sy * cp;
    double rt02 =  sp;
    double rt10 =  sy * cr + cy * sp * sr;
    double rt11 =  cy * cr - sy * sp * sr;
    double rt12 = -cp * sr;
    double rt20 =  sy * sr - cy * sp * cr;
    double rt21 =  cy * sr + sy * sp * cr;
    double rt22 =  cp * cr;

    for (int y = 0; y < h; y++) {
        float *mx = map_x.ptr<float>(y);
        float *my = map_y.ptr<float>(y);
        for (int x = 0; x < w; x++) {
            double lon = (double(x) / w - 0.5) * 2.0 * M_PI;
            double lat = (0.5 - double(y) / h) * M_PI;
            double cl = cos(lat);
            double vx = cl * cos(lon);
            double vy = cl * sin(lon);
            double vz = sin(lat);

            // Apply inverse rotation
            double sx = rt00 * vx + rt01 * vy + rt02 * vz;
            double sy = rt10 * vx + rt11 * vy + rt12 * vz;
            double sz = rt20 * vx + rt21 * vy + rt22 * vz;

            double src_lon = atan2(sy, sx);
            double src_lat = asin(std::max(-1.0, std::min(1.0, (double)sz)));

            // Wrap longitude
            while (src_lon < -M_PI) src_lon += 2.0 * M_PI;
            while (src_lon >  M_PI) src_lon -= 2.0 * M_PI;
            // Clamp latitude
            if (src_lat >  M_PI / 2.0) src_lat =  M_PI / 2.0;
            if (src_lat < -M_PI / 2.0) src_lat = -M_PI / 2.0;

            mx[x] = float((src_lon / (2.0 * M_PI) + 0.5) * w);
            my[x] = float((0.5 - src_lat / M_PI) * h);
        }
    }

    cv::Mat dst;
    cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return dst;
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
        std::cerr << "  -a <int>                Augmented rotations per image (default: 0)\n";
        std::cerr << "  -v, --verbose           Verbose output\n";
        std::cerr << "  -h, --help              Show this help\n\n";
        std::cerr << "Examples:\n";
        std::cerr << "  " << argv[0] << " -s ./SPHORB/ -o vocab.fbow ./pano_images/\n";
        std::cerr << "  " << argv[0] << " -s ./SPHORB/ -o vocab.fbow -k 10 -l 5 -t 8 -n 1000 -a 3 ./city/ ./campus/\n";
        return 1;
    }

    std::vector<std::string> imageDirs;
    std::string sphorbPath;
    std::string outputPath;
    int k = 10, L = 6, nthreads = 1, maxIters = 5, nfeatures = 1000;
    int augment = 0;
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
        } else if (arg == "-a") {
            if (i + 1 < argc) augment = std::stoi(argv[++i]);
            else { std::cerr << "Error: -a requires an integer." << std::endl; return 1; }
        } else if (arg == "--max-iters") {
            if (i + 1 < argc) maxIters = std::stoi(argv[++i]);
            else { std::cerr << "Error: --max-iters requires an integer." << std::endl; return 1; }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg[0] == '-') {
            std::cerr << "Error: unknown option '" << arg << "'." << std::endl;
            return 1;
        } else {
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
    std::set<std::string> imageSet;
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
    std::cout << "Total images: " << images.size()
              << (augment > 0 ? " (x" + std::to_string(augment + 1) + " with augment)" : "")
              << std::endl;

    // Create SPHORB detector.
    std::cout << "Initializing SPHORB detector (nfeatures=" << nfeatures << ")..." << std::endl;
    cv::SPHORB sphorb(sphorbPath, nfeatures, 7, 20);
    std::cout << "  Descriptor size: " << sphorb.descriptorSize() << " bytes ("
              << sphorb.descriptorSize() * 8 << " bits)" << std::endl;

    // Rotation ranges (matching the Python reference)
    const double roll_min  = -M_PI / 6, roll_max  = M_PI / 6;    // ±30°
    const double pitch_min = -M_PI / 6, pitch_max = M_PI / 6;    // ±30°
    const double yaw_min   = -M_PI / 3, yaw_max   = M_PI / 3;    // ±60°

    srand(0);

    // Extract features from all images.
    std::vector<cv::Mat> allFeatures;
    size_t totalKeypoints = 0;
    size_t skipped = 0;
    size_t augmentProcessed = 0;

    for (size_t idx = 0; idx < images.size(); ++idx) {
        std::cout << "[" << (idx + 1) << "/" << images.size() << "] "
                  << images[idx] << std::endl;

        cv::Mat img = cv::imread(images[idx], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "  WARNING: could not read image, skipping." << std::endl;
            ++skipped;
            continue;
        }

        double aspect = static_cast<double>(img.cols) / img.rows;
        if (aspect < 1.5 || aspect > 2.5) {
            std::cerr << "  WARNING: image aspect ratio is " << aspect
                      << " (panoramas usually ~2:1). Results may be suboptimal." << std::endl;
        }

        // Process original image
        {
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sphorb(img, cv::noArray(), keypoints, descriptors);

            if (!descriptors.empty()) {
                allFeatures.push_back(descriptors.clone());
                totalKeypoints += keypoints.size();
            }
            // std::cout << "  orig: " << keypoints.size() << " keypoints" << std::endl;
        }

        // Process augmented rotations
        for (int aug = 0; aug < augment; ++aug) {
            double roll  = roll_min  + (rand() / double(RAND_MAX)) * (roll_max - roll_min);
            double pitch = pitch_min + (rand() / double(RAND_MAX)) * (pitch_max - pitch_min);
            double yaw   = yaw_min   + (rand() / double(RAND_MAX)) * (yaw_max - yaw_min);

            cv::Mat rotated = rotateEquirectangular(img, roll, pitch, yaw);

            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            sphorb(rotated, cv::noArray(), keypoints, descriptors);

            if (!descriptors.empty()) {
                allFeatures.push_back(descriptors.clone());
                totalKeypoints += keypoints.size();
            }
            // std::cout << "  aug" << aug << ": " << keypoints.size()
            //           << " kps  (roll=" << int(roll * 180.0 / M_PI)
            //           << " pitch=" << int(pitch * 180.0 / M_PI)
            //           << " yaw=" << int(yaw * 180.0 / M_PI) << ")" << std::endl;
            ++augmentProcessed;
        }
    }

    if (allFeatures.empty()) {
        std::cerr << "Error: no features extracted from any image." << std::endl;
        return 1;
    }

    size_t imgsProcessed = images.size() - skipped;
    std::cout << "\nSummary:\n";
    std::cout << "  Original images processed: " << imgsProcessed << "/" << images.size() << std::endl;
    std::cout << "  Augmented images processed: " << augmentProcessed << std::endl;
    std::cout << "  Total feature matrices:   " << allFeatures.size() << std::endl;
    std::cout << "  Total keypoints:          " << totalKeypoints << std::endl;
    std::cout << "  Avg kps per extraction:   " << (totalKeypoints / allFeatures.size()) << std::endl;

    // Verify descriptor consistency across images.
    {
        int expectedCols = allFeatures[0].cols;
        int expectedType = allFeatures[0].type();
        for (size_t j = 1; j < allFeatures.size(); ++j) {
            if (allFeatures[j].cols != expectedCols || allFeatures[j].type() != expectedType) {
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

    voc.saveToFile(outputPath);
    std::cout << "Vocabulary saved to " << outputPath << std::endl;

    return 0;
}
