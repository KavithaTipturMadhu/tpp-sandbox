#!/usr/bin/env bash
#
# Runs CMake on the source / build directories

SCRIPT_DIR=$(realpath $(dirname $0))

die_syntax() {
  echo "Syntax: $0 -s SRC_DIR -b BLD_DIR -m MLIR_DIR [-i INST_DIR] [-t (Release|Debug|RelWithDebInfo)] [-c (clang|gcc)] [-g (gcc toolchain)] [-l (ld|lld|gold|mold)] [-S] [-n N]"
  echo ""
  echo "  -i: Optional install dir, default to system"
  echo "  -t: Optional build type flag, default to Release"
  echo "  -c: Optional compiler flag, default to clang"
  echo "  -g: Optional gcc toolchain flag, may be needed by clang"
  echo "  -l: Optional linker flag, default to system linker"
  echo "  -S: Optional sanitizer flag, default to none"
  echo "  -n: Optional link jobs flag, default same as CPUs"
  exit 1
}

# Cmd-line opts
while getopts "s:b:i:m:t:c:g:l:n:S" arg; do
  case ${arg} in
    s)
      SRC_DIR=$(realpath ${OPTARG})
      if [ ! -d ${SRC_DIR}/.git ]; then
        echo "Source '${OPTARG}' not a Git directory"
        die_syntax
      fi
      ;;
    b)
      BLD_DIR=$(realpath ${OPTARG})
      if not mkdir -p ${BLD_DIR}; then
        echo "Error creating build directory '${OPTARG}'"
        die_syntax
      fi
      ;;
    i)
      INST_DIR=$(realpath ${OPTARG})
      if not mkdir -p ${INST_DIR}; then
        echo "Error creating install directory '${OPTARG}'"
        die_syntax
      fi
      ;;
    m)
      MLIR_DIR=$(realpath ${OPTARG})
      if [ ! -f ${MLIR_DIR}/MLIRConfig.cmake ]; then
        echo "MLIR '${OPTARG}' not a CMake directory"
        die_syntax
      fi
      ;;
    g)
      GCC_DIR=$(realpath ${OPTARG})
      GCC_TOOLCHAIN_OPTIONS="-DCMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_DIR} -DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=${GCC_DIR}"
      ;;
    c)
      if [ "${OPTARG}" == "clang" ]; then
        CC=clang
        CXX=clang++
      elif [ "${OPTARG}" == "gcc" ]; then
        CC=gcc
        CXX=g++
      else
        echo "Compiler "${OPTARG}" not recognized"
        die_syntax
      fi
      ;;
    t)
      if [ "${OPTARG}" == "Release" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=${OPTARG}"
      elif [ "${OPTARG}" == "Debug" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=${OPTARG}"
      elif [ "${OPTARG}" == "RelWithDebInfo" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=${OPTARG}"
      else
        echo "Build type "${OPTARG}" not recognized"
        die_syntax
      fi
      ;;
    l)
      if [ "${OPTARG}" == "ld" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DLLVM_USE_LINKER=${OPTARG}"
      elif [ "${OPTARG}" == "lld" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DLLVM_USE_LINKER=${OPTARG}"
      elif [ "${OPTARG}" == "gold" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DLLVM_USE_LINKER=${OPTARG}"
      elif [ "${OPTARG}" == "mold" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DLLVM_USE_LINKER=${OPTARG}"
      else
        echo "Compiler "${OPTARG}" not recognized"
        die_syntax
      fi
      ;;
    S)
      SAN_OPTIONS="-DUSE_SANITIZER=\"Address;Memory;Leak;Undefined\""
      ;;
    n)
      PROCS=$(nproc)
      if [ "${OPTARG}" -gt "0" ] && [ "${OPTARG}" -lt "${PROCS}" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DCMAKE_JOB_POOLS=link=${OPTARG}"
      else
        echo "Invalid value for number of linker jobs '${OPTARG}'"
        die_syntax
      fi
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

# Mandatory arguments
if [ ! "${BLD_DIR}" ] || [ ! "${SRC_DIR}" ] || [ ! "${MLIR_DIR}" ]; then
  die_syntax
fi

if [ ! "${CC}" ] || [ ! "${CXX}" ]; then
  CC=clang
  CXX=clang++
fi

# Check deps
$SCRIPT_DIR/deps.sh

TPP_LIT=$(which lit)

# CMake
cmake -Wno-dev -G Ninja \
    -B${BLD_DIR} -S${SRC_DIR} \
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${INST_DIR} \
    -DMLIR_DIR=${MLIR_DIR} \
    -DLLVM_EXTERNAL_LIT=${TPP_LIT} \
    ${BUILD_OPTIONS} \
    ${GCC_TOOLCHAIN_OPTIONS} \
    ${LINKER_OPTIONS} \
    ${SAN_OPTIONS}
