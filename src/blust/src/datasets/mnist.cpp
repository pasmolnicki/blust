#include <blust/datasets/mnist.hpp>

START_BLUST_NAMESPACE


// Load the images from the file
void mnist::M_load_images(batch_t& images, std::filesystem::path path)
{
	// std::ifstream file(path.wstring(), std::ios::binary);
	std::ifstream file(path.string(), std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("Could not open file: " + path.string());

	int magic_number = 0;
	int n_images	 = 0;
	int n_rows		 = 0;
	int n_cols		 = 0;

	// Read the header
	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&n_images, sizeof(n_images));
	file.read((char*)&n_rows, sizeof(n_rows));
	file.read((char*)&n_cols, sizeof(n_cols));

	// Convert to little-endian
	magic_number = utils::swap_32(magic_number);
	n_images	 = utils::swap_32(n_images);
	n_rows		 = utils::swap_32(n_rows);
	n_cols		 = utils::swap_32(n_cols);

	if (magic_number != MAGIC_NUMBER)
		throw std::runtime_error("Invalid MNIST image file!");

	images.reserve(n_images);

	for (int i = 0; i < n_images; i++)
	{
		// Read the image buffer
		std::vector<uint8_t> image(IMAGE_SIZE);
		file.read((char*)image.data(), IMAGE_SIZE);

		// Create the matrix, and normalize the values
		matrix_t mat_img({ 1, size_t(IMAGE_SIZE) });
		std::transform(image.begin(), image.end(), mat_img.begin(), [](uint8_t c) { return number_t(c) / 255.0f; });
		images.push_back(mat_img);
	}

	file.close();
}

void mnist::M_load_labels(batch_t& labels, std::filesystem::path path)
{
	// std::ifstream file(path.wstring(), std::ios::binary);
	std::ifstream file(path.string(), std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("Could not open file: " + path.string());

	int magic_number	= 0;
	int n_labels		= 0;

	// Read the header
	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&n_labels, sizeof(n_labels));

	// Convert to little-endian
	magic_number = utils::swap_32(magic_number);
	n_labels	 = utils::swap_32(n_labels);

	if (magic_number != LABEL_MAGIC_NUMBER)
		throw std::runtime_error("Invalid MNIST label file!");

	labels.reserve(n_labels);

	for (int i = 0; i < n_labels; i++)
	{
		// Read the label
		char label;
		file.read(&label, 1);

		// Create the one-hot encoded vector
		matrix_t mat_label({ 10, 1 }, 0.0);
		mat_label(label) = 1.0;
		labels.push_back(mat_label);
	}

	file.close();
}

END_BLUST_NAMESPACE