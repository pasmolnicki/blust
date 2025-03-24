project(BlustTests LANGUAGES CXX)

# file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
# add_executable(BlustTests ${TEST_SOURCES})

add_executable(
	BlustTests
	${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
	${CMAKE_CURRENT_SOURCE_DIR}/src/tensor_test.cc
)

target_link_libraries(BlustTests PUBLIC GTest::gtest GTest::gtest_main blust)

include(GoogleTest)
gtest_discover_tests(BlustTests)