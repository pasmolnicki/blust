#pragma once

#include <filesystem>
#include <fstream>
#include <utility>

#include <blust/types.hpp>
#include <blust/settings.hpp>
#include <blust/utils.hpp>

START_BLUST_NAMESPACE

// This is a mnist dataset loader, use it to load the dataset.
// The images are transformed to a 1D vector, with normalized values [0, 1]
// The labels are one-hot encoded (meaning all values are 0, except for the correct label)
class mnist
{
	static void M_load_images(batch_t& images, std::filesystem::path path);
	static void M_load_labels(batch_t& labels, std::filesystem::path path);
public:
	constexpr static const char* IMAGES_FILE		= "train-images.idx3-ubyte";
	constexpr static const char* LABELS_FILE		= "train-labels.idx1-ubyte";
	constexpr static const char* TEST_IMAGES_FILE	= "t10k-images.idx3-ubyte";
	constexpr static const char* TEST_LABELS_FILE	= "t10k-labels.idx1-ubyte";
	constexpr static const int IMAGE_SIZE			= 28 * 28;
	constexpr static const int MAGIC_NUMBER			= 2051;
	constexpr static const int LABEL_MAGIC_NUMBER	= 2049;
	
	// Load the dataset to `images` and `labels` (first, second), 
	// the images are 1D vectors (1x784), and the labels are one-hot encoded
	// If `training` is true, load the training dataset, otherwise load the test dataset
	static std::pair<batch_t, batch_t> load(bool training = true)
	{
		const char* IMAGES_FILE_LOCAL = training ? IMAGES_FILE : TEST_IMAGES_FILE;
		const char* LABELS_FILE_LOCAL = training ? LABELS_FILE : TEST_LABELS_FILE;

		batch_t images, labels;
		auto path = g_settings->path();
		M_load_images(images, (path / IMAGES_FILE_LOCAL));
		M_load_labels(labels, (path / LABELS_FILE_LOCAL));
		return {images, labels};
	}
};

END_BLUST_NAMESPACE