add_library(blust)

target_sources(blust PRIVATE
    src/models/Model.cpp
)

target_include_directories(blust PUBLIC include)