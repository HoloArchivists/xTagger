# xtag: eva (wd EVA02-L) + px (PixAI) CUDA engines, procimg host pipeline. MinGW gcc + CUDA driver API.
# Kernels are compiled by evacc (NVRTC) for the architecture of the GPU visible at build time, then embedded with
# .incbin, so configure / build on the machine that runs it.
#
# From the repository root:
#   cmake -S src -B build -G "MinGW Makefiles"          (options: -DXTAG_CALIB=ON, -DXTAG_WEBP=ON)
#   cmake --build build -j
#   build/xtag src/models/eva02.bin src/models/selected_tags.csv <image|dir>... [-px src/models/pixai.bin]
#
# Layout: xtag.c (CLI)   core/ host pipeline (procimg.h, hp.h)   engine/ CUDA engines + kernels
#         tools/ evacc + model exporters   models/ shipped weights + vocabularies (calib/: fp16 calibration
#         engines and ranges; pixai/: PixAI's configs)   third_party/   docs/
#
# Model prep, once (from the upstream ONNX files, not shipped):
#   python src/tools/export_eva.py model.onnx src/models/calib/eva02_fp16.bin
#   build/xtag_calib src/models/calib/eva02_fp16.bin src/models/selected_tags.csv <images>   -> eva02_fp16.bin.calib
#   python src/tools/export_eva.py model.onnx src/models/eva02.bin src/models/calib/eva02_fp16.bin.calib
#   python src/tools/export_pixai.py model_pixai.onnx src/models/pixai/config_pixai.json ... (same two steps)
cmake_minimum_required(VERSION 3.20)
# MinGW gcc (64-bit): .incbin / -Wa options and the x86-64 intrinsics need it; LLVM's clang may sit earlier on PATH
if(NOT CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    set(CMAKE_C_COMPILER gcc)
endif()
project(xtag LANGUAGES C)

option(XTAG_CALIB "Also build xtag_calib (fp16 engine that records activation ranges)" OFF)
option(XTAG_WEBP "Link libwebp for .webp input" OFF)
option(XTAG_LEAKCHECK "Also build xtag_leak (acquire / release accounting, report at exit)" OFF)
option(XTAG_ORTBENCH "Also build ortbench (stock ONNX Runtime baseline); set ORT_ROOT" OFF)
set(ORT_ROOT "" CACHE PATH "ONNX Runtime GPU package (include/, lib/) for ortbench")

find_package(CUDAToolkit REQUIRED)

# MinGW's platform files seed CMAKE_BUILD_TYPE=Debug before this line runs, so a build-type default never takes: the
# optimisation flags sit on the targets themselves (XT_OPT), whatever the cache says.
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS ON)
set(XT_OPT -O3 -fno-semantic-interposition -fomit-frame-pointer)
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
set(XT_HEADERS ${SRC}/core/procimg.h ${SRC}/core/hp.h ${SRC}/core/steal.h ${ENG}/eva.h ${ENG}/px.h ${ENG}/npu.h)
function(xt_exe target eva_cubin px_cubin)
    add_executable(${target} xtag.c)
    target_include_directories(${target} PRIVATE ${SRC} ${SRC}/core ${ENG})
    target_include_directories(${target} SYSTEM PRIVATE ${CUINC}) # CUDA headers redefine __cdecl under gcc
    target_include_directories(${target} SYSTEM PRIVATE ${SRC}/third_party/level_zero) # NPU (ze_loader.dll at run time)
    target_compile_options(${target} PRIVATE ${XT_OPT} -march=native -Wall -Wextra -Wa,-muse-unaligned-vector-move)
    target_compile_definitions(${target} PRIVATE EVA_CUBIN="${eva_cubin}" PX_CUBIN="${px_cubin}" ${ARGN})
    target_link_directories(${target} PRIVATE ${CUDAToolkit_LIBRARY_DIR})
    target_link_libraries(${target} PRIVATE cuda synchronization) # no cuBLAS: every shipped GEMM is our own kernel
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
    target_link_libraries(xtag_calib PRIVATE cublas) # the fp16 reference engine runs its GEMMs on cuBLAS
    add_dependencies(xtag_calib kernels kernels_calib)
    add_dependencies(xtag kernels_calib) # xtag.c is shared: its objects wait on every cubin it may embed
    list(APPEND XT_CUBINS ${CMAKE_BINARY_DIR}/eva_kernels_calib.cubin)
endif()

# xtag_leak: xtag with every acquire / release wrapped by tools/leakchk.c; prints what is still held after main
if(XTAG_LEAKCHECK)
    xt_exe(xtag_leak ${CMAKE_BINARY_DIR}/eva_kernels.cubin ${CMAKE_BINARY_DIR}/px_kernels.cubin)
    target_sources(xtag_leak PRIVATE tools/leakchk.c)
    target_link_libraries(xtag_leak PRIVATE cublas cublasLt) # leakchk.c's wrappers reference the real symbols
    target_compile_options(xtag_leak PRIVATE -g)
    set(LK_WRAP malloc calloc realloc strdup free fopen fclose
        __imp_CreateFileA __imp_CreateFileMappingA __imp_CreateThread __imp_FindFirstFileA __imp_MapViewOfFile
        __imp_CloseHandle __imp_FindClose __imp_UnmapViewOfFile
        cuMemAlloc_v2 cuMemFree_v2 cuMemAllocHost_v2 cuMemFreeHost cuMemHostRegister_v2 cuMemHostUnregister
        cuEventCreate cuEventDestroy_v2 cuStreamCreate cuStreamDestroy_v2 cuModuleLoadData cuModuleUnload
        cuStreamEndCapture cuGraphDestroy cuGraphInstantiateWithFlags cuGraphExecDestroy
        cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRelease_v2 cublasCreate_v2 cublasDestroy_v2
        cublasLtCreate cublasLtDestroy cublasLtMatmulDescCreate cublasLtMatmulDescDestroy
        cublasLtMatmulPreferenceCreate cublasLtMatmulPreferenceDestroy cublasLtMatrixLayoutCreate
        cublasLtMatrixLayoutDestroy)
    list(TRANSFORM LK_WRAP PREPEND "--wrap=")
    string(REPLACE ";" "," LK_WRAP "${LK_WRAP}")
    target_link_options(xtag_leak PRIVATE "LINKER:${LK_WRAP}")
    add_dependencies(xtag_leak kernels)
endif()

# .incbin is invisible to dependency scanning: rebuild the objects when a cubin or header changes
set_property(SOURCE xtag.c APPEND PROPERTY OBJECT_DEPENDS ${XT_CUBINS} ${XT_HEADERS})

# ---- ortbench: the stock ONNX Runtime baseline (CUDA EP) for the README's comparison
if(XTAG_ORTBENCH)
    add_executable(ortbench tools/ortbench.c)
    target_include_directories(ortbench PRIVATE ${SRC})
    target_include_directories(ortbench SYSTEM PRIVATE ${ORT_ROOT}/include) # SAL macro redefinitions under gcc
    target_compile_options(ortbench PRIVATE ${XT_OPT} -march=native -Wall -Wextra)
    target_link_directories(ortbench PRIVATE ${ORT_ROOT}/lib)
    target_link_libraries(ortbench PRIVATE onnxruntime synchronization)
    foreach(dll onnxruntime onnxruntime_providers_shared onnxruntime_providers_cuda)
        add_custom_command(TARGET ortbench POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_ROOT}/lib/${dll}.dll $<TARGET_FILE_DIR:ortbench>)
    endforeach()
endif()
