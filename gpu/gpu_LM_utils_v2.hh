#pragma once
#include "memory_management.hh"
#include "gpu_search_v2.hh"
#include "../Trie/trie_v2_impl.hh"
#include <sstream>

inline std::vector<unsigned int> allwords (LM &lm);

template<class StringType>
std::pair<bool, std::string> testQueryNgrams(LM& lm, unsigned char * btree_trie_gpu, unsigned int * gpu_first_lvl, StringType arpafile) {
    //Create check against things:
    ArpaReader pesho2(arpafile);
    processed_line text2 = pesho2.readline();

    unsigned int num_keys = 0; //How many ngrams are we going to query
    std::vector<unsigned int> keys;
    std::vector<float> check_against;
    unsigned short max_ngram_order = lm.metadata.max_ngram_order;

    while (!text2.filefinished) {
        //Inefficient reallocation of keys_to_query. Should be done better
        unsigned int num_padded =  max_ngram_order - text2.ngrams.size();

        for (unsigned int i = 0; i < num_padded; i++) {
            text2.ngrams.push_back(0); //Extend ngrams to max num ngrams if they are of lower order
        }
        
        for (unsigned int i = 0; i < max_ngram_order; i++) {
            keys.push_back(text2.ngrams[i]); //Extend ngrams to max num ngrams if they are of lower order
        }

        check_against.push_back(text2.score);

        num_keys++;
        text2 = pesho2.readline();
    }

    unsigned int * gpuKeys = copyToGPUMemory(keys.data(), keys.size());
	float * results;
    allocateGPUMem(num_keys, &results);

    searchWrapper(0, btree_trie_gpu, gpu_first_lvl, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

    //Copy back to host
    float * results_cpu = new float[num_keys];
    copyToHostMemory(results, results_cpu, num_keys);

    //Clear gpu memory
    freeGPUMemory(gpuKeys);
    freeGPUMemory(results);

    bool allcorrect = true;
    std::stringstream error;
    for (unsigned int i = 0; i < num_keys; i++) {
        float res_prob = results_cpu[i];

        float exp_prob = check_against[i];

        if (!(exp_prob == res_prob)) {
            error << "Error expected prob: " << exp_prob << " got: " << res_prob << " at line: " << i << "."<< std::endl;
            allcorrect = false;
            break;
        }
    }
    delete[] results_cpu;

    return std::pair<bool, std::string>(allcorrect, error.str());

}

//Converts a raw sentence into one suitable for generating ngrams from, with vocabIDs
inline std::vector<unsigned int> sent2vocabIDs(LM &lm, std::vector<std::string> input, bool addBeginEndMarkers) {
    std::vector<unsigned int> ret;
    if (addBeginEndMarkers) {
        ret.reserve(input.size() + 2);
    } else {
        ret.reserve(input.size());
    }
    unsigned int unktoken = lm.encode_map.find(std::string("<unk>"))->second; //@TODO don't look up UNKTOKEN every time, get it from somewhere
    unsigned int beginsent = lm.encode_map.find(std::string("<s>"))->second;
    unsigned int endsent = lm.encode_map.find(std::string("</s>"))->second;

    if (addBeginEndMarkers) {
        ret.push_back(beginsent);
    }
    for (auto item : input) {
        std::unordered_map<std::string, unsigned int>::iterator it = lm.encode_map.find(item);
        if (it != lm.encode_map.end()) {
            ret.push_back(it->second);
        } else {
            ret.push_back(unktoken);
        }
    }
    if (addBeginEndMarkers) {
        ret.push_back(endsent);
    }

    return ret;
}

inline std::vector<unsigned int> allwords (LM &lm) {
    std::vector<unsigned int> ret;
    for (std::unordered_map<std::string, unsigned int>::iterator iter = lm.encode_map.begin(); iter != lm.encode_map.end(); iter++  )
    {
        ret.push_back(iter->second);
    }
    return ret;
}

inline std::vector<unsigned int> vocabIDsent2queries(std::vector<unsigned int> vocabIDs, unsigned short ngram_order) {
    std::vector<unsigned int> ret;
	int size = 0;
//	for (int i = ngram_order; i > 1; i--){
	if (vocabIDs.size() < ngram_order){
		size = vocabIDs.size()*vocabIDs.size()-(((vocabIDs.size()-1)*vocabIDs.size())/2);
	}
	else {
	size = (((ngram_order-1)*vocabIDs.size()) - ((ngram_order-1)*(ngram_order-2)/2));
	}
//	}
	ret.reserve(size*ngram_order);
//	ret.reserve(size);
//	printf("???");
	//printf("size is? %d\n",size);
//	printf("waht");
	if (vocabIDs.size()==1){
		ret.push_back(vocabIDs[0]);
		for (int i = 0; i < 4; i++) {
                ret.push_back(0);
        }
	}else if (vocabIDs.size()==2){
		
        ret.push_back(vocabIDs[0]);
        for (int i = 0; i < 4; i++) {
                ret.push_back(0);
        }
		ret.push_back(vocabIDs[1]);
        for (int i = 0; i < 4; i++) {
                ret.push_back(0);
        }
		
		if (ngram_order > 2){
		ret.push_back(vocabIDs[0]);
		ret.push_back(vocabIDs[1]);
        for (int i = 0; i < 3; i++) {
                ret.push_back(0);
        }
		
		}
    }else {

    //In the ret vector put an ngram for every single entry
//	printf("000000000000000");
	for (int distance = 1; distance < ngram_order; distance++){
//		printf("------------------");
		int back_idx = 0;
//		printf("----------------------------------------");
		while (back_idx < (vocabIDs.size()-distance+1)){
			for (int j = 0; j < distance; j++){
				ret.push_back(vocabIDs[j+back_idx]);
				//printf("test %d\n",vocabIDs[j+back_idx]);
			}
			int zeroes_to_pad = ngram_order - distance;
        	for (int i = 0; i < zeroes_to_pad; i++) {
            	ret.push_back(0);
        	}
			back_idx=back_idx+1;
		}
	}}

    return ret;

/*
	if (vocabIDs.size()<ngram_order){
	    ret.reserve((vocabIDs.size() - 1)*ngram_order);
	
	int start = 0;
	while (start < (int)vocabIDs.size()){	

	for (int i = start; i < (int)vocabIDs.size(); i++) {
            ret.push_back(vocabIDs[i]);
        }

    int zeroes_to_pad = ngram_order - (int)vocabIDs.size()+start;
        for (int i = 0; i < zeroes_to_pad; i++) {
            ret.push_back(0);
        }
	start++;
	}

    return ret;
	}else{
        ret.reserve((ngram_order-1)*ngram_order);
		int start = vocabIDs.size()-(ngram_order-1);
    while (start < (int)vocabIDs.size()){

    for (int i = start; i < (int)vocabIDs.size(); i++) {
            ret.push_back(vocabIDs[i]);
        }

    int zeroes_to_pad = ngram_order - (int)vocabIDs.size()+start;
        for (int i = 0; i < zeroes_to_pad; i++) {
            ret.push_back(0);
        }
    start++;
    }

    return ret;
    }*/ // desin for the ole version return no longer than maxngram
}

inline std::vector<std::string> interactiveRead(LM &lm, unsigned char * btree_trie_gpu, unsigned int * gpu_first_lvl, bool addBeginEndMarkers = false) {
    std::string response;
    boost::char_separator<char> sep(" ");
    while (true) {
        getline(std::cin, response);
        if (response == "/end") {
            break;
        }
        std::vector<std::string> sentence;
        boost::tokenizer<boost::char_separator<char> > tokens(response, sep);
        for (auto word : tokens) {
            sentence.push_back(word);
        }
        for (auto item : sentence) {
            std::cout << item << " ";
        }
        std::cout << std::endl << "Now vocabIDs:" << std::endl;
        std::vector<unsigned int> vocabIDs = sent2vocabIDs(lm, sentence, addBeginEndMarkers);

        for (auto item : vocabIDs) {
            std::cout << item << " ";
        }
        std::cout << std::endl << "Now to queries:" << std::endl;
        std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, lm.metadata.max_ngram_order);

        for (auto item : queries) {
            std::cout << item << " ";
        }
        std::cout << std::endl;
        //Now process the sentences
        unsigned int num_keys = queries.size()/lm.metadata.max_ngram_order;
        unsigned int * gpuKeys = copyToGPUMemory(queries.data(), queries.size());
        float * results;

		//Convert allwords into vocab IDs
	    std::vector<unsigned int> allvocabIDs = allwords(lm);

		unsigned int results_size = (num_keys*allvocabIDs.size())+1;
	    allocateGPUMem(results_size, &results);
	    cudaMemset(results, 0, results_size*sizeof(float));

        searchWrapper(allvocabIDs.size(),btree_trie_gpu, gpu_first_lvl, gpuKeys, num_keys, results, lm.metadata.btree_node_size, lm.metadata.max_ngram_order);

        //Copy back to host
		float * results_cpu =new float[results_size];
	    copyToHostMemory(results, results_cpu, results_size);

        freeGPUMemory(gpuKeys);
        freeGPUMemory(results);
        //Copy the results back to CPU and print them
        float sum = 0;
        for (unsigned int i = 1; i < results_size; i++) {
           // sum += results_cpu[i];
			if (results_cpu[i]!=0){
            std::cout << i << " " << results_cpu[i] << " " << std::endl;
			sum++;
			}
        }
        std::cout << std::endl << "Prob sum: " << sum << std::endl;
        delete[] results_cpu;
    }
    return std::vector<std::string>{std::string("pesho")};
}

inline unsigned int sent2QueryVec(std::string& sentence, std::vector<unsigned int>& all_queries, LM& lm, bool addBeginEndMarkers) {
    //Tokenize
    boost::char_separator<char> sep(" ");
    std::vector<std::string> tokenized_sentence;
    boost::tokenizer<boost::char_separator<char> > tokens(sentence, sep);
    for (auto word : tokens) {
        tokenized_sentence.push_back(word);
    }

    //convert to vocabIDs
    std::vector<unsigned int> vocabIDs = sent2vocabIDs(lm, tokenized_sentence, addBeginEndMarkers);

    //Convert to ngram Queries @TODO avoid memory copying here by writing directly into all_queries
    std::vector<unsigned int> queries = vocabIDsent2queries(vocabIDs, lm.metadata.max_ngram_order);
    unsigned int num_queries = queries.size(); //How many queries this sentence has.

    //Now write to the global queries vector
    all_queries.resize(all_queries.size() + num_queries);
    std::memcpy(all_queries.data() + (all_queries.size() - num_queries), queries.data(), num_queries*sizeof(unsigned int));

    //Return the number of queries this sentence has:
    return num_queries;

}

template<class StringType>
void sentencesToQueryVector(std::vector<unsigned int>& queries, std::vector<unsigned int>& sent_lengths, LM& lm, StringType sentsFile, bool addBeginEndMarkers = true) {
    std::ifstream queryFile;
    queryFile.open(sentsFile);

    if (queryFile.fail()) {
        std::cerr << "Failed to open file " << sentsFile << std::endl;
        std::exit(EXIT_FAILURE);
    }

    while (!queryFile.eof()) {
        std::string curr_sent;
        std::getline(queryFile, curr_sent);
        if (curr_sent == "") {
            continue; //Skip empty lines
        }
        //Make this sentence into queries
        unsigned int this_sent_queries = sent2QueryVec(curr_sent, queries, lm, addBeginEndMarkers);
        sent_lengths.push_back(this_sent_queries);
    }
}
