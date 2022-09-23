
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/api/serialization.h"
#include "include/dataset/execute.h"
#include "include/dataset/vision.h"


uint64_t GetTimeMicroSeconds() {
    struct timespec t;
    t.tv_sec = t.tv_nsec = 0;
    clock_gettime(/*CLOCK_REALTIME*/0, &t);
    return (uint64_t)t.tv_sec * 1000000ULL + t.tv_nsec / 1000L;
}
struct stat info;
namespace ms = mindspore;
namespace ds = mindspore::dataset;

std::vector<std::string> GetAllFiles(std::string_view dir_name);
DIR *OpenDir(std::string_view dir_name);
std::string RealPath(std::string_view path);
size_t WriteFile(ms::MSTensor& data, std::string outfile);
ms::MSTensor ReadFile(const std::string &file);
int WriteResult(const std::string& dataFile, const std::vector<ms::MSTensor>& outputs);

int main(int argc, char **argv) {
    // set context
    auto context = std::make_shared<ms::Context>();
    auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
    ascend310_info->SetDeviceID(0);
    ascend310_info->SetPrecisionMode("allow_fp32_to_fp16");
    context->MutableDeviceInfo().push_back(ascend310_info);

    // define model
    std::string ecapa_file = argv[1];
    std::string data_path = argv[2];
    ms::Graph graph;
    ms::Status ret = ms::Serialization::Load(ecapa_file, ms::ModelType::kMindIR, &graph);
    if (ret != ms::kSuccess) {
      std::cout << "Load model failed." << std::endl;
      return 1;
    }
    std::cout << "Load model success." << std::endl;
    ms::Model ConvTasnet;

    // build model
    ret = ConvTasnet.Build(ms::GraphCell(graph), context);
    if (ret != ms::kSuccess) {
      std::cout << "Build model failed." << std::endl;
      return 1;
    }
    std::cout << "Build model success." << std::endl;
    // get model info
    std::vector<ms::MSTensor> model_inputs = ConvTasnet.GetInputs();
    if (model_inputs.empty()) {
      std::cout << "Invalid model, inputs is empty." << std::endl;
      return 1;
    }

    std::vector<std::string> feats = GetAllFiles(data_path);
    uint64_t Time1 = GetTimeMicroSeconds();
    for (const auto &feat_file : feats) {
        // prepare input
        std::vector<ms::MSTensor> outputs;
        std::vector<ms::MSTensor> inputs;

        // read image file and preprocess
        auto feat = ReadFile(feat_file);

        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            feat.Data().get(), feat.DataSize());
        // infer
        ret = ConvTasnet.Predict(inputs, &outputs);
        if (ret != ms::kSuccess) {
            std::cout << "Predict model failed." << std::endl;
            return 1;
        }
        int ret1 = WriteResult(feat_file, outputs);
    }
    uint64_t end = GetTimeMicroSeconds();
    printf("The total run time is: %f ms \n", static_cast<double>(end - Time1) / 1000);
    return 0;
}

std::vector<std::string> GetAllFiles(std::string_view dir_name) {
  struct dirent *filename;
  DIR *dir = OpenDir(dir_name);
  if (dir == nullptr) {
    return {};
  }

  /* read all the files in the dir ~ */
  std::vector<std::string> res;
  while ((filename = readdir(dir)) != nullptr) {
    std::string d_name = std::string(filename->d_name);
    // get rid of "." and ".."
    if (d_name == "." || d_name == ".." || filename->d_type != DT_REG)
      continue;
    res.emplace_back(std::string(dir_name) + "/" + filename->d_name);
  }

  std::sort(res.begin(), res.end());
  return res;
}

DIR *OpenDir(std::string_view dir_name) {
  // check the parameter !
  if (dir_name.empty()) {
    std::cout << " dir_name is null ! " << std::endl;
    return nullptr;
  }

  std::string real_path = RealPath(dir_name);

  // check if dir_name is a valid dir
  struct stat s;
  lstat(real_path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    return nullptr;
  }

  DIR *dir;
  dir = opendir(real_path.c_str());
  if (dir == nullptr) {
    std::cout << "Can not open dir " << dir_name << std::endl;
    return nullptr;
  }
  return dir;
}

std::string RealPath(std::string_view path) {
  char real_path_mem[PATH_MAX] = {0};
  char *real_path_ret = realpath(path.data(), real_path_mem);

  if (real_path_ret == nullptr) {
    std::cout << "File: " << path << " is not exist.";
    return "";
  }

  return std::string(real_path_mem);
}

ms::MSTensor ReadFile(const std::string &file) {
  if (file.empty()) {
    std::cout << "Pointer file is nullptr" << std::endl;
    return ms::MSTensor();
  }

  std::ifstream ifs(file);
  if (!ifs.good()) {
    std::cout << "File: " << file << " is not exist" << std::endl;
    return ms::MSTensor();
  }

  if (!ifs.is_open()) {
    std::cout << "File: " << file << "open failed" << std::endl;
    return ms::MSTensor();
  }

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ms::MSTensor buffer(file, ms::DataType::kNumberTypeFloat32, {1, 301, 80}, nullptr, size);

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buffer.MutableData()), size);
  ifs.close();
  return buffer;
}

int WriteResult(const std::string& dataFile, const std::vector<ms::MSTensor> &outputs) {
    std::string homePath = "./result_Files/";
    const int INVALID_POINTER = -1;
    const int ERROR = -2;
    for (size_t i = 0; i < outputs.size(); ++i) {
        size_t outputSize;
        std::shared_ptr<const void> netOutput = outputs[i].Data();
        outputSize = outputs[i].DataSize();
        int pos = dataFile.rfind('/');
        std::string fileName(dataFile, pos + 1);
        fileName.replace(fileName.find('.'), fileName.size() - fileName.find('.'), '_' + std::to_string(i) + ".bin");
        std::string outFileName = homePath + "/" + fileName;
        FILE *outputFile = fopen(outFileName.c_str(), "wb");
        if (outputFile == nullptr) {
            std::cout << "open result file " << outFileName << " failed" << std::endl;
            return INVALID_POINTER;
        }
        size_t size = fwrite(netOutput.get(), sizeof(char), outputSize, outputFile);
        if (size != outputSize) {
            fclose(outputFile);
            outputFile = nullptr;
            std::cout << "write result file " << outFileName << " failed, write size[" << size <<
                "] is smaller than output size[" << outputSize << "], maybe the disk is full." << std::endl;
            return ERROR;
        }
        fclose(outputFile);
        outputFile = nullptr;
    }
    return 0;
}
