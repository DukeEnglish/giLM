#include "gpu_LM_utils_v2.hh"
#include "lm_impl.hh"
#include <memory>
#include <chrono>
#include <ctime>

int main(int argc, char* argv[]){
    if (argc !=5 && argc != 4 && argc != 3) {
        std::cerr << "Usage:" << std::endl << argv[0] << " path_to_binary_lm_dir path_to_test_file [gpuDeviceID=0] [addBeginEndMarkers_bool=1]" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    bool addBeginEndMarkers = true;
    if (argc == 4) {
        setGPUDevice(atoi(argv[3]));
    }

    if (argc == 5) {
    	setGPUDevice(atoi(argv[3]));
    	addBeginEndMarkers = atoi(argv[4]);
    }
    std::chrono::time_point<std::chrono::system_clock> start, readBinaryLM, memcpyBytearrayStart, memcpyBytearray,
        queryFileIOstart, queryFileIOend, gpuPrepareStart, gpuPrepareEnd, copyBackStart, copyBackEnd, memFreeStart, memFreeEnd;

    start = std::chrono::system_clock::now();

    LM lm(argv[1]); //The read in language model

    readBinaryLM = std::chrono::system_clock::now();
    std::cout << "Read in language model:" << std::endl << lm.metadata << "Loading took: "
        << std::chrono::duration<double>(readBinaryLM - start).count() << " seconds." << std::endl;

    //Copy the LM to the GPU memory: 
    memcpyBytearrayStart = std::chrono::system_clock::now();
    unsigned char * btree_trie_gpu = copyToGPUMemory(lm.trieByteArray.data(), lm.trieByteArray.size());
    unsigned int * first_lvl_gpu = copyToGPUMemory(lm.first_lvl.data(), lm.first_lvl.size());
    memcpyBytearray = std::chrono::system_clock::now();
    std::cout << "Copying the LM to GPU memory took: " << std::chrono::duration<double>(memcpyBytearray - memcpyBytearrayStart).count() << " seconds." << std::endl;

    //Now read in the file and prepare ngrams from it
    queryFileIOstart = std::chrono::system_clock::now();

    std::vector<unsigned int> queries;
    std::vector<unsigned int> sent_lengths;
    sentencesToQueryVector(queries, sent_lengths, lm, argv[2], addBeginEndMarkers);

    queryFileIOend = std::chrono::system_clock::now();
    std::cout << "Preparing the queries took: " << std::chrono::duration<double>(queryFileIOend - queryFileIOstart).count() << " seconds." << std::endl;

    //Copy queries to GPU memory and allocate a results vector
    gpuPrepareStart = std::chrono::system_clock::now();
	
    unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order; //Get how many ngram queries we have to do
    unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
    float * results;


	//Convert allwords into vocab IDs
    std::vector<unsigned int> allvocabIDs = allwords(lm);
	unsigned int results_size = (num_keys*allvocabIDs.size())+1;
	allocateGPUMem(results_size, &results);
    cudaMemset(results, 0, results_size*sizeof(float));

    gpuPrepareEnd = std::chrono::system_clock::now();
    std::cout << "Copying queries to gpu and other gpu work took: " << std::chrono::duration<double>(gpuPrepareEnd - gpuPrepareStart).count() << " seconds." << std::endl;
    
    //Now execute the search
    searchWrapper(allvocabIDs.size(), btree_trie_gpu, first_lvl_gpu, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //copy results back to CPU
    copyBackStart = std::chrono::system_clock::now();

    std::unique_ptr<float[]> results_cpu(new float[results_size]);
    copyToHostMemory(results, results_cpu.get(), results_size);

    copyBackEnd = std::chrono::system_clock::now();
    std::cout << "Copying results to host took: " << std::chrono::duration<double>(copyBackEnd - copyBackStart).count() << " seconds." << std::endl;

    //Free GPU memory
    memFreeStart = std::chrono::system_clock::now();

    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);
    freeGPUMemory(btree_trie_gpu);
    freeGPUMemory(first_lvl_gpu);

    memFreeEnd = std::chrono::system_clock::now();

    std::cout << "Memory free took: " << std::chrono::duration<double>(memFreeEnd - memFreeStart).count() << " seconds." << std::endl;
    //@TODO write a small kernel to sum the scores for each sentence in parallel
    //Print a sum of the total score of the file:
    double sum = 0;
    for (unsigned int i = 0; i < num_keys; i++) {
//        sum += results_cpu[i];
		sum += i;
    }
   // std::cout << "Total file sum is: " << sum << std::endl;
	std::cout << "The number of returns for all queries" << sum << std::endl;
    return 0;
}
