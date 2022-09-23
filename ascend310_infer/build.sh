
if [ ! -d build ]; then
  mkdir build
fi
cd build || exit
cmake .. \
    -DMINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"
make
