#pragma once
#include <vector>
#include <utility>
#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include "structs.hh"
#include <set>
#include <iterator>
#include <sstream>
#include <queue>
#include <assert.h>

class B_tree_node; //Forward declaration

class B_tree {
    private:
        void assign_depth(); //We will only assign depth after compression.

    public:
        B_tree_node * root_node;
        unsigned int size;  //Number of entries in the BTree

        B_tree(unsigned short);
        ~B_tree();
        void draw_tree();
        void insert_entry(Entry value);
        void produce_graph(const char * filename);

        template<class EntryOrVocabID>  //Work with both full entries or just the vocabID
        std::pair<B_tree_node *, int> find_element(EntryOrVocabID element);

        void compress();
        void trim();
        void toByteArray(std::vector<unsigned char>& byte_arr, bool pointer2Index);
        size_t getTotalTreeSize(bool pointer2Index); //Get the total tree size. For preallocating the byte array.
};

class B_tree_node {
    
    public:
        unsigned short max_elem;
        std::vector<Entry> words;
        std::vector<B_tree_node *> children;
        B_tree_node * parent;
        B_tree * container; //Accessible only from the root node
        bool is_root;
        unsigned int depth; //We need this when converting to an array.

        //Operations
        B_tree_node(unsigned short, B_tree_node *);
        B_tree_node(unsigned short, B_tree *);
        ~B_tree_node();
        //Insertion to vector location is index to be inserted (before the old one)
        void insert(B_tree_node * new_node, int location);
        void insert(Entry new_val, int location);

        template<class EntryOrVocabID> //Work with both full entries or just the vocabID
        std::pair<B_tree_node *, int> find_element(EntryOrVocabID element);

        std::pair<B_tree_node *, int> find_position(Entry new_value);
        void split_rebalance();
        bool compress(bool prev_change);
        bool compress_tree();
        void split();
        void split_last();
        void trim();
        void assign_depth(unsigned int what_is_my_depth);

        //For converting to byte array:
        unsigned short getNodeSize(bool pointer2Index);
        void toByteArray(std::vector<unsigned char>& byte_arr, unsigned int children_offset, bool pointer2Index);

};

/*Iterates over a compressed and a non compressed Btree. A better implementation.
Returns last entry of the Btree and sets this.finished to true if we try to increment it
past the size of the Btree.
*/

class Pseudo_btree_iterator {
    private:
        unsigned short current_word; //Current word to return
        unsigned short cur_item; //Finds what child of the parent (in a row) we are.
        B_tree_node * cur_node;
        B_tree * container;
        enum State { NODE, LEAF};
        State state;
        /*
        The different states are:
        1) NODE: We are at an internal node
        3) LEAF: We are at a leaf node
        */
        void find_order(){ //find which child we are.
            for (unsigned short i = 0; i < cur_node->parent->children.size(); i++) {
                if (cur_node->parent->children[i] == cur_node){
                    cur_item = i;
                };
            }
        }

        void up_one() { //Go up one and position at the appropriate item
            while (cur_node->parent) {
                this->find_order();
                cur_node = cur_node->parent;
                current_word = cur_item; //We went up one level so the current word would be the current item.
                //Because len(words) = len(children) - 1
                if (current_word >= cur_node->words.size()){
                    //We are at the end of the node, we need to go to an upper one.
                    continue;
                } else {
                    state = NODE;
                    break;
                }
            }
        }

        void down_as_much_as_possible() { //Given node we go to its leaf
            if (cur_node->children.size() != 0 && cur_node->children[cur_item]) {
                cur_node = cur_node->children[cur_item];
                cur_item = 0; //We are the first child. and we expect the first word
                current_word = 0;
                while (cur_node->children.size() != 0 && cur_node->children.front()) {
                    cur_node = cur_node->children.front();
                }
            }
            if (cur_node->children.size() == 0) {
                state = LEAF;
            } else {
                state = NODE; //We are at a node that contains null children
            }
        }

    public:
        unsigned int current_index = 1;
        bool finished = false; //Keep track if we have finished traversing the trie

        Pseudo_btree_iterator (B_tree_node * root) : current_word(0), cur_item(0), cur_node(root), container(root->container) {
            down_as_much_as_possible(); //Go to the leftmost leaf of the tree.
        }

        unsigned int get_item() {
            return cur_node->words[current_word].value;
        }

        Entry * get_entry() {
            return &cur_node->words[current_word];
        }

        void increment(){
            current_index++;
            if (current_index > container->size) {
                finished = true;
                return; //We have finished iterating through the btree
            }
            switch(state) {
                case LEAF : {
                    current_word++; //We are at a leaf. Just go to the next word if it's available.
                    if (current_word < cur_node->words.size()) {
                        return;
                    } else {
                        up_one();
                        return;
                    }
                    
                }
                case NODE : {
                    B_tree_node * current_node = cur_node; //Keep track of the current node
                    cur_item++; //Go to the next prospective child
                    down_as_much_as_possible(); //If there is no next prospective child (null child), go to next element.
                    if (cur_node == current_node) {
                        current_word++;
                        if (current_word >= cur_node->words.size()) {
                            up_one(); //In case we are at the last element of the node, go up one level.
                        }
                    }
                    return;
                }
            }
                    
        }
};

struct B_tree_node_rec {
//This struct will be used to reconstruct things from the byte array;
//For testing purposes, needs manual deallocation
    bool last;
    unsigned short num_entries;
    unsigned int end_offset; //What is the end offset of this node.
    Entry * entries;
    unsigned int first_child_offset;
    unsigned short * children_sizes;
};

B_tree::B_tree(unsigned short num_max_elem) {
    B_tree_node * root = new B_tree_node(num_max_elem, this);
    root_node = root;
    size = 0;
}

B_tree::~B_tree() {
    B_tree_node * actual_node = root_node;
    delete actual_node;
}

template<class EntryOrVocabID>
std::pair<B_tree_node *, int> B_tree::find_element(EntryOrVocabID element) {
    return root_node->find_element(element);
}

void B_tree::compress(){
    bool has_changed = true;
    while (has_changed) {
        root_node->trim();
        has_changed = root_node->compress_tree();
    }
    //After compression is done, assign depth!
    this->assign_depth();
}

void B_tree::trim(){
    root_node->trim();
}

size_t B_tree::getTotalTreeSize(bool pointer2Index = false) {
    size_t total_size = sizeof(unsigned int); //Size of the first node in the beginning of the tree, 4 bytes
    std::queue<B_tree_node *> node_queue;
    node_queue.push(root_node);
    while (!node_queue.empty()){
        B_tree_node * current_node = node_queue.front();
        total_size += current_node->getNodeSize(pointer2Index);
        for (auto child : current_node->children) {
            if (child) { //Avoid null children
                node_queue.push(child);
            }
        }
        node_queue.pop();
    }
    return total_size;
}

void B_tree::assign_depth(){
    root_node->assign_depth(0);
}

void B_tree::toByteArray(std::vector<unsigned char>& byte_arr, bool pointer2Index = false){
    unsigned int siblings_offset = root_node->getNodeSize(pointer2Index); //Offset because of siblings (including us)
    unsigned int children_offset = sizeof(unsigned int); //Offset because of siblings' offspring and because of 
                                                          // the size of the first node which we will prepend to the byte array
    unsigned int root_size = root_node->getNodeSize(pointer2Index);
    byte_arr.resize(byte_arr.size() + sizeof(root_size));
    std::memcpy(byte_arr.data() + byte_arr.size() - sizeof(root_size), &root_size, sizeof(root_size));

    std::queue<B_tree_node *> processing_queue;
    processing_queue.push(root_node);
    unsigned int current_depth = 0;

    while (!processing_queue.empty()) {
        B_tree_node * current_node = processing_queue.front();
        if (current_depth < current_node->depth) {
            //We have reached a new level. All the children from before are now siblings.
            //The siblings_offset is now added to the children_offset and the children offset is 0
            siblings_offset += children_offset;
            children_offset = 0;
            current_depth++;
        }

        //Convert the node to byte array
        current_node->toByteArray(byte_arr, siblings_offset + children_offset, pointer2Index);

        //Add the children and their offsets to the queue.
        for (auto child : current_node->children){
            if (child) { //Avoid null children
                processing_queue.push(child);
                children_offset += child->getNodeSize(pointer2Index);
            }
        }
        processing_queue.pop();
    }
}

B_tree_node::B_tree_node(unsigned short num_max_elem, B_tree_node * parent_node) {
    max_elem = num_max_elem;
    is_root = false;
    parent = parent_node;
    container = nullptr;
}

B_tree_node::B_tree_node(unsigned short num_max_elem, B_tree * container_orig) {
    max_elem = num_max_elem;
    is_root = true;
    parent = nullptr;
    container = container_orig;
}

B_tree_node::~B_tree_node(){
    words.clear();
    for (size_t i = 0; i < children.size(); i++) {
        if (children[i]) {
            delete children[i]; //Prevent nullptr free. @TODO is this necessary?
        }
    }
}

void B_tree_node::assign_depth(unsigned int what_is_my_depth){
    this->depth = what_is_my_depth;
    for (auto child : children) {
        if (child) {
            child->assign_depth(what_is_my_depth + 1);
        }
    }
}

unsigned short B_tree_node::getNodeSize(bool pointer2Index){
    /*Size is:
    getEntrySize(pointer2Index)*len(words) + len(children)*sizeof(unsigned short) (Size of each child)
    + sizeof(unsigned int) (offset to the first child)
    unless we are at a bottom most node len(children) = len(words)*/
    unsigned short entry_size = getEntrySize(pointer2Index);
    if (children.size() != 0) {
        return entry_size*words.size() + sizeof(unsigned short)*children.size() + sizeof(unsigned int);
    } else {
        return entry_size*words.size();
    }
    
}

void B_tree_node::toByteArray(std::vector<unsigned char>& byte_arr, unsigned int children_offset, bool pointer2Index = false){
    /*Appends to the byte_array the contents of this node in the following order:
    (first-child-offset (unsigned int))((children)-offsets (unsigned short))(words-in-byte-array)
    the children_offsets parameter will tell us how many elements there are going to be between us and our first child.
    If we don't have children, we just contain the words array*/
    unsigned int offset = byte_arr.size(); //Keep track of offset in order to do direct memory copies without using .push_back()
    unsigned int last;
    //Set last
    if (children.size() == 0) {
        last = true;
    } else {
        last = false;
    }

    if (!last) {
    //We only need to do this if we actually have children.

        //Determine how much to change the size of the byte_arr:
        byte_arr.resize(byte_arr.size() + sizeof(unsigned int) + children.size()*sizeof(unsigned short));

        //The offset of the first child needs to be stored now. It is  just the children_offset that we have.
        std::memcpy(byte_arr.data() + offset, &children_offset, sizeof(children_offset));
        offset += sizeof(children_offset);

        //Get the size of each child
        unsigned short childrensize[children.size()];
        for (size_t i = 0; i < children.size(); i++) {
            if (children[i]){
                childrensize[i] = children[i]->getNodeSize(pointer2Index);
            } else {
                childrensize[i] = 0; //If we have a null child its size is 0
            }
        }

        //Push it onto the byte array
        std::memcpy(byte_arr.data() + offset, &childrensize[0], sizeof(unsigned short)*children.size());
    }

    //Copy the words
    EntriesToByteArray(byte_arr, words, pointer2Index);
}

void B_tree_node::insert(B_tree_node * new_node, int location){
    std::vector<B_tree_node *>::iterator it = children.begin() + location;
    children.insert(it, new_node);
}

void B_tree_node::insert(Entry new_val, int location){
    std::vector<Entry>::iterator it = words.begin() + location;
    words.insert(it, new_val);
}

template<class EntryOrVocabID>
std::pair<B_tree_node *, int> B_tree_node::find_element(EntryOrVocabID element) {

    int candidate_position = words.size(); //Assume last position
    bool found = false;

    for (size_t i = 0; i < words.size(); i++){
        if (words[i] == element) {
            candidate_position = i;
            found = true;
            break;
        }
        if (words[i] > element) {
            //We can never have two nodes with the same value as per specification.
            candidate_position = i;
            break;
        }
    }

    if (children.size() != 0 && !found){
        return children[candidate_position]->find_element(element);
    } else {
        if (found){
            return std::pair<B_tree_node *, int>(this, candidate_position);
        } else {
            return std::pair<B_tree_node *, int>(nullptr, -1); //Not found, nullptr
        }
    }

}

std::pair<B_tree_node *, int> B_tree_node::find_position(Entry new_value) {

    int candidate_position = words.size(); //Assume last position

    for (size_t i = 0; i < words.size(); i++){
        if (words[i] > new_value) {
            //We can never have two nodes with the same value as per specification.
            candidate_position = i;
            break;
        }
    }

    if (children.size() != 0){
        return children[candidate_position]->find_position(new_value);
    } else {
        return std::pair<B_tree_node *, int>(this, candidate_position);
    }

}

void B_tree_node::split_rebalance() {
    if (words.size() <= max_elem){
        //No need to split the node. It's balanced.
        return;
    } else {
        split();
        if (parent) {
            parent->split_rebalance(); //Sometimes we don't have a parent
        }
    }
}

void B_tree_node::trim() {
    //Trims the B_tree so that no empty nodes are left
    bool all_null = true; //Check if all children are nullpointers
    for (size_t i = 0; i<children.size(); i++) {
        if (children[i] && children[i]->words.size() == 0) {
            delete children[i];
            children[i] = nullptr;
        }
    }

    for (auto child : children) {
        if (child) {
            child->trim();
            all_null = false; //We accessed a child so it can't be null
        }
    }

    if (all_null){
        children.clear(); //Clear all children if everything is null.
    }
}

bool B_tree_node::compress_tree(){
    bool has_modified_this = false; //Check if we need to call compress again
    bool has_modified_child = false; //Check if any of the children have modified
    bool has_modified_grandchild = false; //Check if any of the grandchildren have modified

    if (this->words.size() < max_elem) {
        has_modified_this = this->compress(false);
    }

    for (auto child : children) {
        if (!child) {
            continue;
        }
        if (child->words.size() < max_elem){
            has_modified_child = child->compress(false);
        }
        has_modified_grandchild = child->compress_tree();
        //Check if any change happened at all.
        has_modified_this = has_modified_child || has_modified_grandchild || has_modified_this;
    }

    return has_modified_this;
}

bool B_tree_node::compress(bool prev_change) {
    //If a node doesn't have a full number of children, move some entries up

    //Find which children have the most number of elements. If more than one
    //Choose randomly between the two
    std::vector<B_tree_node *> max_child; //A vector that will contain the children with maximum nodes
    size_t max_children = 0; //Max children so far

    //Get a vector of all child nodes that contain the greatest amount of children
    for (auto child : children) {
        if (!child) {
            continue;
        }
        if (max_children < (child->words.size())) { //We have a node with more elements than anything we've seen before
            max_children = child->words.size();
            max_child.clear();
            max_child.push_back(child);
        } else if (max_children == child->words.size() && (child->words.size() != 0)) {
            max_child.push_back(child);
        }
    }

    //If we have no suitable candidates we don't split.
    if (max_child.size() == 0) {
        return prev_change;
    }
    //Choose in between the suitable candidates
    B_tree_node * childtosplit = max_child[(size_t)(rand() % max_child.size())];
    //Don't split a single child if it has children. They will populate it.
    //Don't split a child with two words, because one will move up and the other one will 
    //be left empty and we might lose its children if they are non empty. Only split a child
    //with two words if they have no children.
    //@TODO. This check should be moved upwards
    if (childtosplit->words.size() < 3 && childtosplit->children.size() > 0) {
        return prev_change;
    }
    //Reccursively compress until either all children are sized 1, or we are saturated.
    childtosplit->split();
    prev_change = true;
    if (this->words.size() < max_elem) {
        prev_change = this->compress(prev_change);
    }
    return prev_change;
}

void B_tree_node::split_last(){
    //Bottom most node
    int new_location = parent->words.size();
    for (size_t i = 0; i< parent->words.size(); i++) {
        if (parent->words[i] > words[0]) {
            new_location = i;
            break;
        }
    }
    //insert the middle_value and the right node (the left was there beforehand)
    parent->insert(words[0], new_location);
    //Insert dummy child on its place
    B_tree_node * dummy_child = nullptr;
    parent->insert(dummy_child, new_location+1);
    //This node will be destroyed by the trim function
    
    words.clear();
    children.clear(); //Destroy any dangling children.
}

void B_tree_node::split() {
    int middle_idx = words.size()/2; //Integer division here, always right


    if (middle_idx == 0) {
        //We are trying to split a node with only 1 element. Instead move it to the parent.
        if (this->children.size() == 0) { //Only do that if we are bottom most
            this->split_last();
        }
        return;
    }

    //We if we need to split we take our current node to become the left node
    //by trimming it and we will create a new node which will be the right node
    //from the elements that we previously cut out.

    //Calculate middle index for children. Different cases for odd and even
    //We can't use (children.size() + 1)/2 because children.size() can be 0
    int child_mid_idx;
    if (children.size() % 2 == 0) {
        //Even words:
        child_mid_idx = children.size()/2;
    } else {
        child_mid_idx = (children.size()/2) + 1;
    }

    //Save the middle value;
    Entry middle_value = words[middle_idx];


    //Create the right node
    B_tree_node * right_node = new B_tree_node(max_elem, parent);

    //populate it.
    for (std::vector<Entry>::iterator it = words.begin() + middle_idx + 1; it != words.end(); it++){
        right_node->words.push_back(*it);
    }
    
    for (std::vector<B_tree_node *>::iterator it = children.begin() + child_mid_idx; it != children.end(); it++){
        if (*it) {
            (*it)->parent = right_node; //Skip null children
        }
        right_node->children.push_back(*it);
    }

    //Trim the left node.
    this->words.resize(middle_idx);
    this->children.resize(child_mid_idx);

    //Assign parent node and change root if necessary
    if (parent == nullptr) {
        //We are the root node, we need to make a new root.
        B_tree_node * new_root = new B_tree_node(max_elem, container);
        new_root->insert(middle_value, 0);

        this->is_root = false;
        new_root->container = this->container;
        new_root->container->root_node = new_root;
        this->container = nullptr;
        this->parent = new_root;
        right_node->parent = new_root;

        //Now assign child pointers to the new root.
        new_root->insert(right_node, 0);
        new_root->insert(this, 0);
        //We are done, the tree is balanced and split;
    } else {
        //Find the location of the middle_value in the parent
        int new_location = parent->words.size();
        for (size_t i = 0; i< parent->words.size(); i++) {
            if (parent->words[i] > middle_value) {
                new_location = i;
                break;
            }
        }
        //insert the middle_value and the right node (the left was there beforehand)
        parent->insert(middle_value, new_location);
        parent->insert(right_node, new_location+1);

    }

}

void B_tree::insert_entry(Entry value) {
    B_tree_node * root = root_node;
    std::pair<B_tree_node *, int> position = root->find_position(value);
    position.first->insert(value, position.second);
    position.first->split_rebalance();
    size++;
}

inline void print_node(B_tree_node * node) {
    std::cout << std::endl << "Node_ID: " << node << std::endl;
    if (node->parent) {
        std::cout << "Parent is: " << node->parent;
    }
    std::cout << " Num keys: " << node->words.size();
    std::cout << " Num children: " << node-> children.size();
    std::cout << " words: ";
    for (unsigned short i = 0; i<node->words.size(); i++) {
        std::cout << node->words[i].value << ' ';
    }
    for (unsigned short i = 0; i<node->children.size(); i++) {
        print_node(node->children[i]);
    }

}

void B_tree::draw_tree() {
    //Draws what is currently in the tree. For debugging purposes
    print_node(root_node);
}

inline void draw_node(B_tree_node * node, std::ofstream& filehandle, int num_child) {

    filehandle << "node" << node << "[label = \"<f0> ";
    for (size_t i = 0; i<node->words.size(); i++) {
        filehandle << '|' << node->words[i].value << '|' << "<f" << (i + 1) << '>';
    }
    filehandle << "\"];\n";
    if (num_child != -1) {
        //Root has children but has no parents so it has no incoming connections
        //Otherwise draw the connection
        filehandle << "\"node" << node->parent << "\":f" <<  num_child  << " -> \"node" << node << "\"\n";
    }
    for (size_t i = 0; i < node->children.size(); i++) {
        if (node->children[i]) {
            draw_node(node->children[i], filehandle, i); //Due to compression we may have nullptr child.
        }
    }
}

void B_tree::produce_graph(const char * filename) {
    std::ofstream graphfile;
    graphfile.open(filename);
    graphfile << "digraph g {\nnode [shape = record,height=.1];\n";
    draw_node(root_node, graphfile, -1);
    graphfile << "}\n";
    graphfile.close();
}

inline void B_tree_node_reconstruct(B_tree_node_rec& target, std::vector<unsigned char>& byte_arr, unsigned int start, unsigned short size,
    unsigned short max_btree_node_size, bool pointer2Index = false){
    //Populates the given B_tree_node_reconstruct knowing the byte_arr, the start index and the size of the first item.
    bool last;
    unsigned short entry_size = getEntrySize(pointer2Index);

    //Determine if the node is last or not. If the node is internal it is always going to be full because of our compression.
    /*Equation is num_entries = (size - first_child_offset - last_entry_size)/x*(unsigned short (for the remaining entries)
        + sizeof(word))*/
    if (((size - sizeof(unsigned int) - sizeof(unsigned short)) / (entry_size + sizeof(unsigned short))) == max_btree_node_size) {
        last = false;
        assert(((size - sizeof(unsigned int) - sizeof(unsigned short)) % (entry_size + sizeof(unsigned short))) == 0);
    } else {
        last = true;
        assert((size%getEntrySize(pointer2Index)) == 0);
    }
    target.last = last;

    if (!last) {
        //Calculate the indexes knowing the structure of the byte array:
        //(first-child-offset (unsigned int))((children)-offsets (unsigned short))(words-in-byte-array)
        unsigned short num_entries = max_btree_node_size;

        //Get the first child offset (in absolute terms, from the start of the array);
        memcpy((unsigned char *)&target.first_child_offset, byte_arr.data() + start, sizeof(unsigned int));

        //Get the sizes of each child
        unsigned short * children_sizes = new unsigned short[num_entries+1];
        memcpy((unsigned char *)children_sizes, byte_arr.data() + start + sizeof(unsigned int),
            sizeof(unsigned short)*(num_entries + 1));
        target.children_sizes = children_sizes;

        //Construct the entries vector
        unsigned int entries_start_offset = start + sizeof(unsigned int) + (num_entries+1)*sizeof(unsigned short);
        target.entries = byteArrayToEntries(byte_arr, num_entries, entries_start_offset, pointer2Index);
        target.num_entries = num_entries;
    } else {
        //We only have entries
        unsigned short num_entries = size/getEntrySize(pointer2Index);

        unsigned int entries_start_offset = start;
        target.entries = byteArrayToEntries(byte_arr, num_entries, entries_start_offset, pointer2Index);
        target.num_entries = num_entries;
    }
}

template<class EntryOrVocabID>  //Work with both full entries or just the vocabID
std::pair<bool, Entry> search_byte_arr(std::vector<unsigned char>& byte_arr, EntryOrVocabID key, unsigned short max_btree_node_size,
    bool pointer2Index = false, unsigned int this_btree_start = 0) {
    //For testing purposes. Tries to find a key in the byte array. This will basically test if
    //our byte array has been stored properly. Uses simple linear search cause I can't be bothered
    //The this_btree_start argument is used if the byte array contains more than one btrees.

    //First, find the size of the root node:
    unsigned int root_size;
    memcpy((unsigned char *)&root_size, byte_arr.data() + this_btree_start, sizeof(root_size));


    bool found = false;
    Entry found_entry;
    B_tree_node_rec temp_node = {false, 0, 0, nullptr, 0, nullptr};
    unsigned int start = sizeof(root_size);
    unsigned short size = root_size;
    while (!found && !temp_node.last) {
        B_tree_node_reconstruct(temp_node, byte_arr, start + this_btree_start, size, max_btree_node_size, pointer2Index);

        for (unsigned short i = 0; i < temp_node.num_entries; i++){
            if (temp_node.entries[i].value == key) {
                found = true;
                found_entry = temp_node.entries[i];
                break;
            } else if (temp_node.entries[i].value > key){
                //Calculate the next start end indexes 
                start = temp_node.first_child_offset;
                for (unsigned short j = 0; j < i; j++){
                    start+= temp_node.children_sizes[j];
                }
                size = temp_node.children_sizes[i];
                break;
            } else if ((temp_node.entries[i].value < key) && (i == temp_node.num_entries - 1)) {
                //If we have reached the last entry but the key is still bigger, go to the right node
                start = temp_node.first_child_offset;
                for (unsigned short j = 0; j < i; j++) {
                    start+= temp_node.children_sizes[j];
                }
                start+= temp_node.children_sizes[i];
                size = temp_node.children_sizes[i+1];
                break;    
            }
        }
        //Don't leak memory
        delete[] temp_node.entries;
        if (!temp_node.last) {
            delete[] temp_node.children_sizes; //We'd get a double free otherwise
        }
    }
    return std::pair<bool, Entry>(found, found_entry);

}

inline std::pair<bool, std::string> test_btree_array(std::set<unsigned int>& input, std::vector<unsigned char>& byte_arr,
    unsigned short max_btree_node_size, bool pointer2Index = false) {
    bool passes = true;
    int counter = 0;
    std::stringstream error;

    for (std::set<unsigned int>::iterator it = input.begin(); it != input.end(); it++) {
        Entry key = {*it, nullptr, 0.0, 0.0};
        std::pair<bool, Entry> res = search_byte_arr(byte_arr, key, max_btree_node_size, pointer2Index);
        if (!res.first) {
            passes = false;
            error << "ERROR! " << *it << " Not found at position: " << counter << std::endl;
        }
        counter++;
    }
    return std::pair<bool, std::string>(passes, error.str());
}


inline std::pair<bool, std::string> test_btree(std::set<unsigned int> &input, B_tree * tree) {

    B_tree_node * root_node = tree->root_node;
    Pseudo_btree_iterator * iter = new Pseudo_btree_iterator(root_node);
    bool passes = true;
    int counter = 0;
    std::stringstream error;

    for (std::set<unsigned int>::iterator it = input.begin(); it != input.end(); it++) {
        if (*it != iter->get_item()) {
            passes = false;
            error << "ERROR! Expected: " << *it << " Got: " << iter->get_item() << " At position: " << counter << std::endl;
            break;
        }
        counter++;
        iter->increment();
    }
    //Test if the iterator recognizes that it has finished the btree
    iter->increment();
    if (!iter->finished) {
        passes = false;
        error << "Iterator has not reported to have finished yet!" << std::endl;
    }
    delete iter;
    return std::pair<bool, std::string>(passes, error.str());
}
