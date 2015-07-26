//
// This code creates DataShard for RNNLM dataset.
// It is adapted from the convert_mnist_data from Caffe
// The RNNLM dataset could be downloaded at
//    http://www.rnnlm.org/
//
// Usage:
//    create_shard.bin input_filename output_foldername

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <fstream>
#include <string>
#include <map>
#include <vector>

#include "utils/data_shard.h"
#include "utils/common.h"
#include "proto/common.pb.h"

using singa::DataShard;

using StrIntGreaterMap = std::map<std::string, int, std::greater<std::string> >;


void create_shard(const char *input, const char *classOutput, const char *wordOutput, int nclass) {
    // load input file
    std::ifstream in(input);
    CHECK(in) << "Unable to open file " << input;

    // calc word's frequency, sort by map
    std::string word;
    StrIntGreaterMap wordFreqMap;
    while (in >> word) {
        // TODO(kaiping): improve tokenize logic for complex input format (such as symbols)
        ++wordFreqMap[word];
    }
    nwords = wordFreqMap.size();

    int sumFreq = 0;
    for (auto it : wordFreqMap) {
        sumFreq += it->second;
    }

    // index words after sorting, and split words into classes by freq
    std::vector<std::pair<int, int> > classInfo;
    StrIntGreaterMap wordIdxMap, wordClassIdxMap;
    int classIdxCnt = 0;
    int tmpWordFreqSum = 0;
    int nextStartPos = 0;
    int wordIdxCnt = 0;
    for (auto it : wordFreqMap) {
        // index words
        wordIdxMap[it->first] = wordIdxMap.size();

        // generate classes
        tmpWordFreqSum += it->second;
        wordClassIdxMap[it->first] = classIdxCnt;

        // split a new class
        // ensure no empty class
        if ((tmpWordFreqSum >= (classIdxCnt + 1) * sumFreq / nclass) || (nwords - wordIdxCnt) <= nclass - classIdxCnt) {
            classInfo.emplace_back(std::make_pair(nextStartPos, wordIdxCnt));
            nextStartPos = wordIdxCnt + 1;
            ++classIdxCnt;
        }

        ++wordIdxCnt;
    }

    // generate class data
    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    DataShard classShard(classOutput, DataShard::kCreate);
    singa::Record record;
    record.set_type(singa::Record::kWordClass);
    singa::WordClassRecord *classRecord = record.mutable_class_record();
    for (int i = 0; i != classInfo.size(); ++i) {
        classRecord->set_start(classInfo[i].first);
        classRecord->set_end(classInfo[i].second);
        classRecord->set_class_index(i);
        snprintf(key, kMaxKeyLength, "%08d", i);
        classShard.Insert(std::string(key), record);
    }
    classShard.Flush();
    record.clear_class_record();

    // generate word data
    // reset input stream status for second loading
    in.clear();
    in.seekg(0, std::ios_base::beg);
    DataShard wordShard(wordOutput, DataShard::kCreate);
    record.set_type(singa::Record::kSingleWord);
    singa::SingleWordRecord *wordRecord = record.mutable_word_record();
    int wordStreamCnt = 0;
    while (in >> word) {
        // TODO (kaiping): do not forget here if modify tokenize logic
        auto classIdxIter = wordClassIdxMap.find(word);
        if (wordClassIdxMap.end() == iter) continue;
        wordRecord->set_name(word);
        wordRecord->set_word_index(wordIdxMap[word]);
        wordRecord->set_class_index(classIdxIter->second);
        snprintf(key, kMaxKeyLength, "%08d", i++);
        wordShard.Insert(std::string(key), record);
    }
    wordShard.Flush();
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "This program create a DataShard for a RNNLM dataset\n"
                "The RNNLM dataset could be downloaded at\n"
                "    http://www.rnnlm.org/\n"
                "You should gunzip them after downloading.\n"
                "Usage:\n"
                "    create_shard.bin input_file output_class_path, output_word_path, class_size";
    } else {
        google::InitGoogleLogging(argv[0]);
        int classSize = atoi(argv[4]);
        CHECK(classSize) << "class size parse failed. [" << argv[4] << "]";
        create_shard(argv[1], argv[2], argv[3], classSize);
    }
    return 0;
}
