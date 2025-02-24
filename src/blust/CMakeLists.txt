add_library(blust)

target_sources(blust PRIVATE
	"src/models/Model.cpp"
	"src/cuda/cuda_driver.cpp"
)

target_link_libraries(blust PUBLIC CUDA::cuda_driver)

target_include_directories(blust PUBLIC include)