#include "trie.hh"
#include "memory_management.hh"
#include "gpu_search.hh"
#include <chrono>
#include <ctime>

int main(int argc, char* argv[]) {
    LM lm;
    createTrieArray(argv[1], atoi(argv[2]), lm);
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    //input ngram
    std::string response;
    boost::char_separator<char> sep(" ");
    while (true) {
        getline(std::cin, response);
        if (response == "/end") {
            break;
        }

        std::vector<unsigned int> keys_to_query;
        boost::tokenizer<boost::char_separator<char> > tokens(response, sep);
        for (auto word : tokens) {
            keys_to_query.push_back(std::stoul(word, 0, 10));
        }

        unsigned int num_keys = 1; //How many ngrams are we going to query

        unsigned int * gpuKeys = copyToGPUMemory(keys_to_query.data(), keys_to_query.size());
        float * results;
        allocateGPUMem(num_keys, &results);

        unsigned int entrySize = getEntrySize(/*pointer2index =*/ true);
        searchWrapper(btree_trie_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, entrySize, lm.metadata.max_ngram_order);

        //Copy back to host
        float * results_cpu = new float[num_keys];
        copyToHostMemory(results, results_cpu, num_keys);

        std::cout << "Query result is: " << results_cpu[0] << std::endl;

        freeGPUMemory(gpuKeys);
        freeGPUMemory(results);

        delete[] results_cpu;
    }

    freeGPUMemory(btree_trie_gpu);
}
