#pragma once

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
	
	// Load the dataset to `images` and `labels`, the images are 1D vectors (1x784), and the labels are one-hot encoded
	static void mnist::load_dataset(batch_t& images, batch_t& labels)
	{
		auto path = g_settings->path();
		M_load_images(images, (path / IMAGES_FILE));
		M_load_labels(labels, (path / LABELS_FILE));
	}

	// Load the training dataset to `images` and `labels` the images are 1D vectors (1x784), and the labels are one-hot encoded
	static void mnist::load_training(batch_t& images, batch_t& labels)
	{
		auto path = g_settings->path();
		M_load_images(images, (path / TEST_IMAGES_FILE));
		M_load_labels(labels, (path / TEST_LABELS_FILE));
	}
};

END_BLUST_NAMESPACE