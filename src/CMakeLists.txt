# xtag: eva (wd EVA02-L) + px (PixAI) CUDA engines, procimg host pipeline. MinGW gcc + CUDA driver API.
# Kernels are compiled by evacc (NVRTC) for the architecture of the GPU visible at build time, then embedded with
# .incbin, so configure / build on the machine that runs it.
#
# From the repository root:
#   cmake -S src -B build -G "MinGW Makefiles"          (options: -DXTAG_CALIB=ON, -DXTAG_WEBP=ON)
#   cmake --build build -j
#   build/xtag src/models/eva02.bin src/models/selected_tags.csv <image|dir>... [-px src/models/pixai.bin]
#
# Layout: xtag.c (CLI)   core/ host pipeline (procimg.h, hp.h) + model conversion scripts   engine/ CUDA engines
#         + kernels   tools/ evacc, pxcal, ortbench   models/ tag lists, PixAI configs; converted weights land here
#         (not shipped)   third_party/   docs/
#
# Converting the models (README: Converting the models), with -DXTAG_CALIB=ON:
#   python src/core/export_eva.py model.onnx src/models/calib/eva02_fp16.bin
#   build/xtag_calib src/models/calib/eva02_fp16.bin src/models/selected_tags.csv <images>   -> eva02_fp16.bin.calib
#   python src/core/export_eva.py model.onnx src/models/eva02.bin src/models/calib/eva02_fp16.bin.calib
#   python src/core/export_pixai.py model_pixai.onnx src/models/pixai/config_pixai.json src/models/calib/pixai_fp16.bin
#   python src/core/pxfeed.py <images> | build/pxcal src/models/calib/pixai_fp16.bin       -> pixai_fp16.bin.calib
#   python src/core/export_pixai.py model_pixai.onnx src/models/pixai/config_pixai.json src/models/pixai.bin
#          src/models/calib/pixai_fp16.bin.calib
cmake_minimum_required(VERSION 3.20)
# MinGW gcc (64-bit): .incbin / -Wa options and the x86-64 intrinsics need it; LLVM's clang may sit earlier on PATH
if(NOT CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    set(CMAKE_C_COMPILER gcc)
endif()
project(xtag LANGUAGES C)

option(XTAG_CALIB "Also build xtag_calib (fp16 engine that records activation ranges)" OFF)
option(XTAG_WEBP "Link libwebp for .webp input" OFF)
option(XTAG_ORTBENCH "Also build ortbench (stock ONNX Runtime baseline); set ORT_ROOT" OFF)
set(ORT_ROOT "" CACHE PATH "ONNX Runtime GPU package (include/, lib/) for ortbench")

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(SRC ${CMAKE_SOURCE_DIR})
set(ENG ${SRC}/engine)
set(CUINC ${CUDAToolkit_INCLUDE_DIRS})
list(GET CUINC 0 CUINC0)

# ---- kernel compiler (build-time tool)
add_executable(evacc tools/evacc.c)
target_compile_options(evacc PRIVATE -O2 -Wall -Wextra)
target_include_directories(evacc SYSTEM PRIVATE ${CUINC})
target_link_directories(evacc PRIVATE ${CUDAToolkit_LIBRARY_DIR})
target_link_libraries(evacc PRIVATE cuda nvrtc)

# cubin <name> <source .cu> [nvrtc options...] -> ${CMAKE_BINARY_DIR}/<name>.cubin
set(KDEPS ${ENG}/kern_common.cuh)
function(xt_cubin name cu)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/${name}.cubin
        COMMAND evacc ${ENG}/${cu} ${CMAKE_BINARY_DIR}/${name}.cubin ${CUINC0} -I${ENG} ${ARGN}
        DEPENDS evacc ${ENG}/${cu} ${KDEPS}
        COMMENT "evacc ${cu} -> ${name}.cubin"
        VERBATIM)
endfunction()

xt_cubin(eva_kernels eva_kernels.cu -DEVA_FP8)
xt_cubin(px_kernels px_kernels.cu -DEVA_FP8)
add_custom_target(kernels DEPENDS ${CMAKE_BINARY_DIR}/eva_kernels.cubin ${CMAKE_BINARY_DIR}/px_kernels.cubin)

# ---- xtag
set(XT_HEADERS ${SRC}/core/procimg.h ${SRC}/core/hp.h ${ENG}/eva.h ${ENG}/px.h)
function(xt_exe target eva_cubin px_cubin)
    add_executable(${target} xtag.c)
    target_include_directories(${target} PRIVATE ${SRC} ${SRC}/core ${ENG})
    target_include_directories(${target} SYSTEM PRIVATE ${CUINC}) # CUDA headers redefine __cdecl under gcc
    target_compile_options(${target} PRIVATE -march=native -Wall -Wextra -Wa,-muse-unaligned-vector-move)
    target_compile_definitions(${target} PRIVATE EVA_CUBIN="${eva_cubin}" PX_CUBIN="${px_cubin}" ${ARGN})
    target_link_directories(${target} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    target_link_libraries(${target} PRIVATE cuda cublas cublasLt synchronization)
    if(XTAG_WEBP)
        target_compile_definitions(${target} PRIVATE PI_WEBP)
        target_link_libraries(${target} PRIVATE webp)
    endif()
endfunction()

xt_exe(xtag ${CMAKE_BINARY_DIR}/eva_kernels.cubin ${CMAKE_BINARY_DIR}/px_kernels.cubin)
add_dependencies(xtag kernels)
set(XT_CUBINS ${CMAKE_BINARY_DIR}/eva_kernels.cubin ${CMAKE_BINARY_DIR}/px_kernels.cubin)

if(XTAG_CALIB)
    xt_cubin(eva_kernels_calib eva_kernels.cu)
    xt_cubin(px_kernels_calib px_kernels.cu)
    add_custom_target(kernels_calib DEPENDS ${CMAKE_BINARY_DIR}/eva_kernels_calib.cubin ${CMAKE_BINARY_DIR}/px_kernels_calib.cubin)
    xt_exe(xtag_calib ${CMAKE_BINARY_DIR}/eva_kernels_calib.cubin ${CMAKE_BINARY_DIR}/px_kernels.cubin EVA_CALIB)
    add_dependencies(xtag_calib kernels kernels_calib)
    add_dependencies(xtag kernels_calib) # xtag.c is shared: its objects wait on every cubin it may embed
    list(APPEND XT_CUBINS ${CMAKE_BINARY_DIR}/eva_kernels_calib.cubin)

    # pxcal: PixAI calibration (fp16 px engine fed by core/pxfeed.py on stdin)
    add_executable(pxcal tools/pxcal.c)
    target_include_directories(pxcal PRIVATE ${SRC} ${ENG})
    target_include_directories(pxcal SYSTEM PRIVATE ${CUINC})
    target_compile_options(pxcal PRIVATE -march=native -Wall -Wextra -Wa,-muse-unaligned-vector-move)
    target_compile_definitions(pxcal PRIVATE PX_CALIB EVA_CUBIN="${CMAKE_BINARY_DIR}/eva_kernels.cubin"
                                             PX_CUBIN="${CMAKE_BINARY_DIR}/px_kernels_calib.cubin")
    target_link_directories(pxcal PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    target_link_libraries(pxcal PRIVATE cuda cublas cublasLt synchronization)
    add_dependencies(pxcal kernels kernels_calib)
    set_property(SOURCE tools/pxcal.c APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_BINARY_DIR}/px_kernels_calib.cubin
                 ${CMAKE_BINARY_DIR}/eva_kernels.cubin ${ENG}/px.h ${ENG}/eva.h)
endif()

# .incbin is invisible to dependency scanning: rebuild the objects when a cubin or header changes
set_property(SOURCE xtag.c APPEND PROPERTY OBJECT_DEPENDS ${XT_CUBINS} ${XT_HEADERS})

# ---- ortbench: the stock ONNX Runtime baseline (CUDA EP) for the README's comparison
if(XTAG_ORTBENCH)
    add_executable(ortbench tools/ortbench.c)
    target_include_directories(ortbench PRIVATE ${SRC})
    target_include_directories(ortbench SYSTEM PRIVATE ${ORT_ROOT}/include) # SAL macro redefinitions under gcc
    target_compile_options(ortbench PRIVATE -march=native -Wall -Wextra)
    target_link_directories(ortbench PRIVATE ${ORT_ROOT}/lib)
    target_link_libraries(ortbench PRIVATE onnxruntime synchronization)
    foreach(dll onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)
        add_custom_command(TARGET ortbench POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_ROOT}/lib/${dll}.dll $<TARGET_FILE_DIR:ortbench>)
    endforeach()
endif()
