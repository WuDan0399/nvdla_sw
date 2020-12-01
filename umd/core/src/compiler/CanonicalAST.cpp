/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <algorithm>
#include <string>

#include "priv/Check.h"

#include "priv/CanonicalAST.h"

#include "priv/Network.h"
#include "priv/Tensor.h"
#include "ErrorMacros.h"

#include "rapidjson/document.h"

using std::map;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::string;
using std::endl;
using std::ostream;
using std::stringstream;

namespace nvdla
{

class ILayer;

namespace priv
{

ENUM_PARAMETER_STATIC(canonical_ast::CanonicalOpType,  CANONICAL_OPERATION_TYPE_ENUMS,  "CanonicalOpTypeEnum")

NvU32 canonical_ast::Node::m_next_id = 0;
NvU32 canonical_ast::Edge::m_next_id = 0;

DataType canonical_ast::getDataType(const char * name){
    NvDlaError e = NvDlaSuccess;
    if (!strcmp(name, "float32")){
        return DataType::FLOAT;
    }
    else{
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Data Type");
    }
fail:
    return e;
}
DataFormat canonical_ast::getDataFormate(const char * name){
    NvDlaError e = NvDlaSuccess;
    if (!strcmp(name, "NCHW")){
        return DataFormat::NCHW;
    }
    else if (!strcmp(name, "NHWC")){
        return DataFormat::NHWC;
    }
    else{
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unsupported Data Format");
    }
fail:
    return e;
}

int canonical_ast::getDims2FromValue(const rapidjson::Value& a, vector<Dims2 *> b)
{
    // I need an assert to identify whether size of a is 2*n
    for (rapidjson::SizeType i = 0; i < a.Size()/2; i++) // Uses SizeType instead of size_t
    {
        NvS32 s, t;
        std::stringstream strValue;
        strValue << a[2*i].GetString();
        strValue >> s;
        b[i]->h = s;
        strValue << a[2*i+1].GetString();
        strValue >> t;
        b[i]->w = t;
    }
    return 0;
}

LayerType canonical_ast::gettypeFromJson(rapidjson::Value::ConstMemberIterator itr){
    std::string name(itr->name.GetString());
    if(name.find("pool")!=std::string::npos){
        return LayerType::kPOOLING;
    }
    else if(name.find("conv")!=std::string::npos){
        return LayerType::kCONVOLUTION;
    }
    else if(name.find("dense")!=std::string::npos){
        return LayerType::kFULLY_CONNECTED;
    }
    else if(name.find("relu")!=std::string::npos){
        return LayerType::kACTIVATION;
    }
    else if (name.find("tanh") != std::string::npos){
        return LayerType::kACTIVATION;
    }
    else if (name.find("sigmoid") != std::string::npos){
        return LayerType::kACTIVATION;
    }
    else if (name.find("softmax") != std::string::npos){
        return LayerType::kSOFTMAX;
    }
    else{
        gLogError << "unrecognized layer type"<<endl;
    }
}

canonical_ast::Node *canonical_ast::newCanonicalNodeFromJson(rapidjson::Value::ConstMemberIterator itr)
{
    LayerType original_type = canonical_ast::gettypeFromJson(itr);

    switch (original_type)
    {
        case LayerType::kCONVOLUTION: {
            return canonical_ast::NodeFactory::newConvNodeFromJson(itr);
          }
        case LayerType::kFULLY_CONNECTED: {
            return canonical_ast::NodeFactory::newFCNodeFromJson(itr);
          }
        case LayerType::kACTIVATION: {
            return canonical_ast::NodeFactory::newActivationNodeFromJson(itr);
          }
        case LayerType::kPOOLING: {
            return canonical_ast::NodeFactory::newPoolingNodeFromJson(itr);
          }
        // case LayerType::kLRN: {
        //     return canonical_ast::NodeFactory::newLRNNode(lrn_layer);
        //   }
        // case LayerType::kSCALE: {
        //     return canonical_ast::NodeFactory::newScaleNode(scale_layer);
        //   }
        // case LayerType::kBATCH_NORM: {
        //     return canonical_ast::NodeFactory::newBatchNormNode(bn_layer);
        //   }
        case LayerType::kSOFTMAX: {
            return canonical_ast::NodeFactory::newSoftMaxNodeFromJson(itr);
          }
        // case LayerType::kCONCATENATION: {
        //     return canonical_ast::NodeFactory::newConcatNode(concat_layer);
        //   }
        // case LayerType::kDECONVOLUTION: {
        //     return canonical_ast::NodeFactory::newDeconvNode(deconv_layer);
        //   }
        // case LayerType::kELEMENTWISE: {
        //     return canonical_ast::NodeFactory::newEWNode(ew_layer);
        //   }
        // case LayerType::kSLICE: {
        //     return canonical_ast::NodeFactory::newSplitNode(slice_layer);
        //   }
        default:
            return NULL;
    }

    return NULL;
}

canonical_ast::Node *canonical_ast::newCanonicalNode(Layer *orig_nw_layer)
{
    LayerType original_type = orig_nw_layer->getType();

    switch (original_type)
    {
        case LayerType::kCONVOLUTION: {
            ConvolutionLayer *conv_layer = LayerFactory::derivedPriv< ConvolutionLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newConvNode(conv_layer);
          }
        case LayerType::kFULLY_CONNECTED: {
            FullyConnectedLayer *fc_layer = LayerFactory::derivedPriv< FullyConnectedLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newFCNode(fc_layer);
          }
        case LayerType::kACTIVATION: {
            ActivationLayer *act_layer = LayerFactory::derivedPriv< ActivationLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newActivationNode(act_layer);
          }
        case LayerType::kPOOLING: {
            PoolingLayer *pool_layer = LayerFactory::derivedPriv< PoolingLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newPoolingNode(pool_layer);
          }
        case LayerType::kLRN: {
            LRNLayer *lrn_layer = LayerFactory::derivedPriv< LRNLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newLRNNode(lrn_layer);
          }
        case LayerType::kSCALE: {
            ScaleLayer *scale_layer = LayerFactory::derivedPriv< ScaleLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newScaleNode(scale_layer);
          }
        case LayerType::kBATCH_NORM: {
            BatchNormLayer *bn_layer = LayerFactory::derivedPriv< BatchNormLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newBatchNormNode(bn_layer);
          }
        case LayerType::kSOFTMAX: {
            SoftMaxLayer *sm_layer = LayerFactory::derivedPriv< SoftMaxLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newSoftMaxNode(sm_layer);
          }
        case LayerType::kCONCATENATION: {
            ConcatenationLayer *concat_layer = LayerFactory::derivedPriv< ConcatenationLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newConcatNode(concat_layer);
          }
        case LayerType::kDECONVOLUTION: {
            DeconvolutionLayer *deconv_layer = LayerFactory::derivedPriv< DeconvolutionLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newDeconvNode(deconv_layer);
          }
        case LayerType::kELEMENTWISE: {
            ElementWiseLayer *ew_layer = LayerFactory::derivedPriv< ElementWiseLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newEWNode(ew_layer);
          }
        case LayerType::kSLICE: {
            SliceLayer *slice_layer = LayerFactory::derivedPriv< SliceLayerDiamond >( orig_nw_layer );
            return canonical_ast::NodeFactory::newSplitNode(slice_layer);
          }
        default:
            return NULL;
    }

    return NULL;
}

//
// the following generates a 1:1 mapping with the Canonical graph input.
//
//This function is writen by Dan Wu
//


canonical_ast::Graph *canonical_ast::generateGraphFromJson(rapidjson::Document* doc)
{
    vector<canonical_ast::Edge *> input_edges;
    vector<canonical_ast::Edge *> output_edges;

    // map<canonical_ast::Node *,std::string, Graph::nodeCompareFn>  node_name;
    // map<canonical_ast::Node *, std::string, Graph::nodeCompareFn>::iterator lni;
    map<std::string, canonical_ast::Node *> name_node;
    map<std::string, std::string> id_name;
    map<std::string, canonical_ast::Node *>::iterator lni;

    map<Tensor *, canonical_ast::Edge *>  tensor_edge;
    map<Tensor *, Tensor *>  nw_tensor_to_can_tensor;
    map<Tensor *, canonical_ast::Edge *>::iterator tei;

    Graph *graph = new Graph();
    
    /*get number of input, then resize the input_edges*/
    input_edges.resize((*doc)["input"]["dest"].GetArray().Size());

    /*get number of output, then resize the output_edges*/
    //output_edges.resize(0);

    // preprocess, add bias to objects
    for (rapidjson::Value::ConstMemberIterator itr = doc->MemberBegin();itr != doc->MemberEnd(); ++itr)
    {
        std::string name(itr->name.GetString());
        id_name[itr->value.GetObject()["id"].GetString()]=name;
    }
    rapidjson::Document::AllocatorType& a = doc->GetAllocator();
    for (rapidjson::Value::ConstMemberIterator itr = doc->MemberBegin();itr != doc->MemberEnd(); ++itr)
    {
        std::string name(itr->name.GetString());
        if(name.find("bias")!=std::string::npos)
        {
            std::string src_name = id_name[itr->value["source"][0].GetString()];
            if(! doc->HasMember(src_name.c_str()))
            {
                gLogError << "Detached bias layer (" << name <<") with source node ("<< src_name <<")"<<endl;
            }
            else{
                //Add new member to this object
                // change dest
                (*doc)[src_name.c_str()]["dest"].Clear();
                for( int i = 0; i < itr->value["dest"].GetArray().Size(); i++)
                {
                    std::string dest_name = itr->value.GetObject()["dest"][i].GetString();
                    rapidjson::Value m_dest;
                    m_dest.SetString(dest_name.c_str(), a);
                    (*doc)[src_name.c_str()]["dest"].PushBack( m_dest ,a);
                }
                // add other bias info as an Object
                rapidjson::Value bias_info = rapidjson::Value((*doc)[name.c_str()], a);
                (*doc)[src_name.c_str()].AddMember("bias_info", bias_info, a);
            }
        }
        if(name.find("flatten")!=std::string::npos){
            std::string src_name = id_name[itr->value["source"][0].GetString()];
            std::string dest_name = id_name[itr->value["dest"][0].GetString()];
            assert(doc->HasMember(src_name.c_str()));
            assert(doc->HasMember(dest_name.c_str()));

            //Change dest node weight shape
            if( !(*doc)[dest_name.c_str()].HasMember("weight_shape")){
                gLogError << "No weight shape for node: " << dest_name <<endl;
            }
            int ori_n = (*doc)[dest_name.c_str()]["weight_shape"][0].GetInt();
            int ori_c = (*doc)[dest_name.c_str()]["weight_shape"][1].GetInt();
            
            if( (*doc)[src_name.c_str()]["shape"][0][0].Size() != 4 ){
                gLogError << "Not acceptable shape for node (" << src_name <<") followed by batch_flatten node (" <<name <<")" <<endl;
            }
            int new_n = (*doc)[src_name.c_str()]["shape"][0][0][0].GetInt();
            int new_c = (*doc)[src_name.c_str()]["shape"][0][0][1].GetInt();
            int new_h = (*doc)[src_name.c_str()]["shape"][0][0][2].GetInt();
            int new_w = (*doc)[src_name.c_str()]["shape"][0][0][3].GetInt();
            
            (*doc)[dest_name.c_str()]["weight_shape"].Clear();
            rapidjson::Value m_n(ori_n), m_c(new_c), m_h(new_h), m_w(new_w);
            (*doc)[dest_name.c_str()]["weight_shape"].PushBack(m_n ,a);
            (*doc)[dest_name.c_str()]["weight_shape"].PushBack(m_c ,a);
            (*doc)[dest_name.c_str()]["weight_shape"].PushBack(m_h ,a);
            (*doc)[dest_name.c_str()]["weight_shape"].PushBack(m_w ,a);
            
            // Change dest and source of before and after nodes
            // TODO: polish this paragraph, not neat enough
            // save dest for source node
            vector<std::string> dest_links;
            int dest_size = (*doc)[src_name.c_str()]["dest"].GetArray().Size();
            for (int i=0; i<dest_size; i++){
                std::string dest_link = (*doc)[src_name.c_str()]["dest"].GetArray()[i].GetString();
                dest_links.push_back(dest_link);
            }
            //rewrite and change the dest with name batch_flatten
            (*doc)[src_name.c_str()]["dest"].Clear();
            for (int i=0; i<dest_size; i++){
                if (dest_links[i].find("flatten")!=std::string::npos){
                    std::string m = dest_links[i];
                    rapidjson::Value n;
                    n.SetString(m.c_str(), a);
                    (*doc)[src_name.c_str()]["dest"].PushBack(n, a);
                }
                else{
                    std::string m = itr->value["dest"][0].GetString();
                    rapidjson::Value n;
                    n.SetString(m.c_str(), a);
                    (*doc)[src_name.c_str()]["dest"].PushBack(n, a);
                }
            }

            // save source for dest node
            vector<std::string> src_links;
            int src_size = (*doc)[dest_name.c_str()]["source"].GetArray().Size();
            for (int i=0; i<src_size; i++){
                std::string src_link = (*doc)[dest_name.c_str()]["source"].GetArray()[i].GetString();
                src_links.push_back(src_link);
            }
            (*doc)[dest_name.c_str()]["source"].Clear();
            for (int i=0; i<src_size; i++){
                if(src_links[i].find("flatten")!=std::string::npos){
                    std::string m = src_links[i];
                    rapidjson::Value n;
                    n.SetString(m.c_str(), a);
                    (*doc)[dest_name.c_str()]["source"].PushBack(n, a);
                }
                else{
                    std::string m = itr->value["source"][0].GetString();
                    rapidjson::Value n;
                    n.SetString(m.c_str(), a);
                    (*doc)[dest_name.c_str()]["source"].PushBack(n, a);
                }
            }

        }
    }

    /*generate nodes*/
    for (rapidjson::Value::ConstMemberIterator itr = doc->MemberBegin();itr != doc->MemberEnd(); ++itr)
    {
        std::string name(itr->name.GetString());
        if(name=="input"){ continue;}
        if(name.find("bias")!=std::string::npos){continue;}
        if(name.find("flatten")!=std::string::npos){continue;}

        canonical_ast::Node *can_node = newCanonicalNodeFromJson(itr);
        if ( !can_node )
        {
            delete graph; // blow up
            graph = 0;
            return graph;
        }
        can_node->setGraph(graph);
        graph->insertNode(can_node);

        can_node->setId(graph->nextNodeId());
        can_node->setName(name);
        name_node[name] = can_node;  
    }

    // Insert the first edge, input to the netwpork
    Tensor input_tensor;
    input_tensor.setDimensions(Dims4((*doc)["input"]["shape"][0][0][0].GetInt(), 
        (*doc)["input"]["shape"][0][0][1].GetInt(), 
        (*doc)["input"]["shape"][0][0][2].GetInt(), 
        (*doc)["input"]["shape"][0][0][3].GetInt()));
    input_tensor.setDataType(getDataType((*doc)["input"]["dtype"][0][0].GetString()));
    std::vector<NvF32> chnlScales;
    for(int i =0; i< (*doc)["input"]["shape"][0][0][1].GetInt(); i++){
        chnlScales.push_back(1);
    }
    input_tensor.setChannelScales(chnlScales);
    for(int i =0; i < (*doc)["input"]["dest"].GetArray().Size(); ++i){
        canonical_ast::Edge * can_edge = new canonical_ast::Edge();
        can_edge->setGraph(graph);
        Tensor* can_tensor = input_tensor.clone();
        can_tensor->setNetwork(NULL);   // get rid of any connections back to the network builder
        can_tensor->setTensorType(TensorType::kNW_INPUT);
        can_edge->setId(graph->nextEdgeId());
        can_edge->setOriginalTensor(can_tensor);
        graph->insertEdge(can_edge);
        // Find edge_side and node
        ast::EdgeSide edge_side(ast::EdgeSideEnum::SECOND);
        ast::EdgeDirection edge_dir(ast::EdgeDirectionEnum::DIRECTED);
        canonical_ast::Node * dest_node = name_node[id_name[(*doc)["input"]["dest"][i].GetString()]];
        graph->appendNodeToEdge(can_edge, edge_side, dest_node);
        dest_node->markInputEdge(can_edge);
        //mark input
        input_edges[i]=can_edge;
	}
    //Insert edges for each layer
    for (lni=name_node.begin(); lni!=name_node.end(); ++lni){
        std::string name = lni->first;
        canonical_ast::Node *m_node = lni->second;
        Tensor m_tensor;
        switch ((*doc)[name.c_str()]["shape"][0][0].GetArray().Size()){
            case 2:
                m_tensor.setDimensions(Dims4((*doc)[name.c_str()]["shape"][0][0][0].GetInt(), 
                (*doc)[name.c_str()]["shape"][0][0][1].GetInt(), 
                1, 1));
                break;
            case 4:
                m_tensor.setDimensions(Dims4((*doc)[name.c_str()]["shape"][0][0][0].GetInt(), 
                (*doc)[name.c_str()]["shape"][0][0][1].GetInt(), 
                (*doc)[name.c_str()]["shape"][0][0][2].GetInt(), 
                (*doc)[name.c_str()]["shape"][0][0][3].GetInt()));
                break;
            default:
                gLogError<<"Unsupported output shape for node ("<< name <<")"<<endl;
        }
        m_tensor.setDataType(getDataType((*doc)[name.c_str()]["dtype"][0][0].GetString()));
        std::vector<NvF32> chnlScales;
        for(int i =0; i< (*doc)[name.c_str()]["shape"][0][0][1].GetInt(); i++){
            chnlScales.push_back(127);
        }
        m_tensor.setChannelScales(chnlScales);
        if((*doc)[name.c_str()].HasMember("layout")){
            m_tensor.setDataFormat(getDataFormate((*doc)[name.c_str()]["layout"][0][0].GetString()));
            }
        else if((*doc)[name.c_str()].HasMember("data_layout")){
            m_tensor.setDataFormat(getDataFormate((*doc)[name.c_str()]["data_layout"][0][0].GetString()));
            }
        /////////////////////////////////TODO HERE/////////////////
        if((*doc)[name.c_str()]["dest"].GetArray().Size()==0){
            //This edge should be the output edge
            canonical_ast::Edge * can_edge = new canonical_ast::Edge();
            can_edge->setGraph(graph);
            Tensor* can_tensor = m_tensor.clone();
            can_tensor->setNetwork(NULL);   // get rid of any connections back to the network builder
            can_tensor->setTensorType(TensorType::kNW_OUTPUT);
            can_edge->setId(graph->nextEdgeId());
            can_edge->setOriginalTensor(can_tensor);
            graph->insertEdge(can_edge);
            ast::EdgeSide edge_side_this(ast::EdgeSideEnum::FIRST);
            graph->appendNodeToEdge(can_edge, edge_side_this, m_node);
            output_edges.push_back(can_edge);
            m_node->markOutputEdge(can_edge);
        }


        for(int i =0; i < (*doc)[name.c_str()]["dest"].GetArray().Size(); ++i){
            canonical_ast::Node * dest_node = name_node[id_name[(*doc)[name.c_str()]["dest"][i].GetString()]];
            canonical_ast::Edge * can_edge = new canonical_ast::Edge();
            can_edge->setGraph(graph);
            Tensor* can_tensor = m_tensor.clone();
            can_tensor->setNetwork(NULL);   // get rid of any connections back to the network builder
            can_tensor->setTensorType(TensorType::kIO);
            can_edge->setId(graph->nextEdgeId());
            can_edge->setOriginalTensor(can_tensor);
            graph->insertEdge(can_edge);

            ast::EdgeSide edge_side_this(ast::EdgeSideEnum::FIRST);
            ast::EdgeSide edge_side_next(ast::EdgeSideEnum::SECOND);
            ast::EdgeDirection edge_dir(ast::EdgeDirectionEnum::DIRECTED);

            graph->appendNodeToEdge(can_edge, edge_side_this, m_node);
            graph->appendNodeToEdge(can_edge, edge_side_next, dest_node);
            m_node->markOutputEdge(can_edge);
            dest_node->markInputEdge(can_edge);
	    }
  
    }


    /*Final Process*/
    if ( input_edges.size() )
    {
        graph->setInputEdges(input_edges);
    }
    if ( output_edges.size() )
    {
        graph->setOutputEdges(output_edges);
    }

    graph->scoredOrdering()->generate();
    graph->markClean();
    return graph;
}


canonical_ast::Graph* canonical_ast::Graph::clone()
{
    REPORT_ERROR(NvDlaError_NotSupported, "Graph cloning is not supported for Canonical AST");

    return NULL;
}

//
// the following generates a 1:1 mapping with the Canonical graph input.
//
canonical_ast::Graph *canonical_ast::generateGraph(Network *network)
{
    vector<canonical_ast::Edge *> input_edges;
    vector<canonical_ast::Edge *> output_edges;

    map<canonical_ast::Node *, Layer *, Graph::nodeCompareFn>  node_layer;
    map<canonical_ast::Node *, Layer *, Graph::nodeCompareFn>::iterator lni;

    map<Tensor *, canonical_ast::Edge *>  tensor_edge;
    map<Tensor *, Tensor *>  nw_tensor_to_can_tensor;
    map<Tensor *, canonical_ast::Edge *>::iterator tei;

    Graph *graph = new Graph();

    vector<Tensor *> network_inputs;
    for (int ni = 0; ni < network->getNumInputs(); ++ni)
    {
        network_inputs.push_back(TensorFactory::priv(network->getInput(ni)));
    }
    input_edges.resize(network_inputs.size());


    vector<Tensor *> network_outputs;
    for (int ni = 0; ni < network->getNumOutputs(); ++ni)
    {
        network_outputs.push_back(TensorFactory::priv(network->getOutput(ni)));
    }
    output_edges.resize(network_outputs.size());

    //    gLogInfo << "canonical_ast::" << __func__ << " network shows " << network_inputs.size() << " inputs and " <<
    //        network_outputs.size() << " outputs" << endl;

    for (int li = 0; li < network->getNumLayers(); li++)
    {
        ILayer *ilayer = network->getLayer(li);
        Layer *layer = LayerFactory::priv(ilayer);
        if ( !(ilayer && layer) )
        {
            gLogError << __func__ << " encountered null layer at network layer index=" << li << endl;
            continue;
        }

        canonical_ast::Node *can_node = newCanonicalNode(layer);
        if ( !can_node )
        {
            delete graph; // blow up
            graph = 0;
            goto done;
        }
        can_node->setGraph(graph);
        graph->insertNode(can_node);

        can_node->setId(graph->nextNodeId());
        can_node->setName(layer->getName());

        node_layer[can_node] = layer;
    }

    //
    // Now all the layer nodes are in the graph.
    // For each layer assemble the edges.
    //

    for (lni = node_layer.begin(); lni != node_layer.end(); ++lni)
    {
        canonical_ast::Node *node = lni->first;
        Layer *l = lni->second;

        size_t input_tensors = 0, output_tensors = 0, aux_input_tensors = 0;
        vector<Tensor *> io_tensors, aux_tensors;
        NVDLA_UNUSED(aux_input_tensors);

        for(int ii = 0, II = l->getNumInputs(); ii < II; ++ii)
        {
            Tensor *tensor = TensorFactory::priv(l->getInput(ii));
            if ( !tensor )
            {
                gLogError << __func__ << " 3.<null>.i." << ii << endl;
                continue;
            }
            io_tensors.push_back(tensor);
            input_tensors++;
        }
        for(int oo = 0, OO = l->getNumOutputs(); oo < OO; ++oo)
        {
            Tensor *tensor = TensorFactory::priv(l->getOutput(oo));
            if ( ! tensor )
            {
                gLogError << __func__ << " 3.<null>.o." << oo << endl;
                continue;
            }
            io_tensors.push_back(tensor);
            output_tensors++;
        }

        for(size_t io = 0, IO = io_tensors.size(); io < IO; ++io)
        {
            Tensor *nw_tensor = io_tensors[io];
            bool is_input = io < input_tensors;
            ast::EdgeSide edge_side( is_input ? ast::EdgeSideEnum::SECOND : ast::EdgeSideEnum::FIRST);
            ast::EdgeDirection edge_dir(ast::EdgeDirectionEnum::DIRECTED);

            map<Tensor *, canonical_ast::Edge *>::iterator f = tensor_edge.find(nw_tensor);
            canonical_ast::Edge *can_edge = 0;
            Tensor* can_tensor = 0;
            if ( f == tensor_edge.end() )
            {
                can_edge = new canonical_ast::Edge();
                can_edge->setGraph(graph);

                can_tensor = nw_tensor->clone();
                can_tensor->setNetwork(NULL);   // get rid of any connections back to the network builder
                can_tensor->setTensorType(TensorType::kIO);
                can_edge->setId(graph->nextEdgeId());
                can_edge->setOriginalTensor(can_tensor);
                graph->insertEdge(can_edge);

                tensor_edge[nw_tensor] = can_edge;
                nw_tensor_to_can_tensor[nw_tensor] = can_tensor;
            } else {
                can_edge = f->second;
            }
            graph->appendNodeToEdge(can_edge, edge_side, node);

            // if this is an input node it could be one of the network inputs.
            // if so keep track of it.
            if ( is_input )
            {
                for ( size_t iti = 0; iti < network_inputs.size(); iti++)
                {
                    if ( nw_tensor == network_inputs[iti] )
                    {
                        // gLogInfo << " identified input edge: " << (int)iti << " tensor id " << tensor->getName() << endl;
                        input_edges[iti] = can_edge;
                        can_tensor = nw_tensor_to_can_tensor[nw_tensor];
                        can_tensor->setTensorType(TensorType::kNW_INPUT);
                        break;
                    }
                }
                node->markInputEdge(can_edge);
            }
            else
            {
                for ( size_t oti = 0; oti < network_outputs.size(); oti++)
                {
                    if ( nw_tensor == network_outputs[oti] )
                    {
                        // gLogInfo << " identified output edge: " << (int)oti << " tensor id " << tensor->getName() << endl;
                        output_edges[oti] = can_edge;
                        can_tensor = nw_tensor_to_can_tensor[nw_tensor];
                        can_tensor->setTensorType(TensorType::kNW_OUTPUT);
                        break;
                    }
                }
                node->markOutputEdge(can_edge);
            }
        }
    }

    if ( input_edges.size() )
    {
        graph->setInputEdges(input_edges);
    }
    if ( output_edges.size() )
    {
        graph->setOutputEdges(output_edges);
    }

    graph->scoredOrdering()->generate();
    graph->markClean();

done:
    return graph;
}

ostream &canonical_ast::outputJson(canonical_ast::Graph *graph, ostream &os)
{
    string sep;
    os << "[ {" << " \"app\" : \"\"}  " << endl; // to signal content flavor

    //
    // nodes
    //
    sep = string(",");
    for (Graph::NodeSetIterator ni = graph->nodes().begin(); ni != graph->nodes().end(); ++ni)
    {
        os << sep;
        outputJson(graph, *ni, os);
        sep = string(", ");
    }

    //
    // edges
    //
    for (Graph::EdgeSetIterator ei = graph->edges().begin(); ei != graph->edges().end(); ++ei)
    {
        os << sep;
        outputJson(graph, *ei, os);
        sep = string(",");
    }
    os << "]" << endl;
    return os;
}



ostream &canonical_ast::outputJson(canonical_ast::Graph *graph, canonical_ast::Edge *edge, ostream &os)
{
#if 0
    static int dummy = 0;
    string edge_text_id = edge->originalTensor()->getName();

    if ( edge_text_id == string("") )
    {
        edge->originalTensor()->setName( string("e-"+toString(dummy++)).c_str());
        edge_text_id = edge->originalTensor()->getName();
    }
#endif
    // edge label to gather the fan-in (almost always 1) and fan-out
    os << "{ \"class\":\"edge\", \"id\":\"" << edge->id() <<
        "\", \"is_input\":" << ( (graph->inputEdges().end()==std::find(graph->inputEdges().begin(), graph->inputEdges().end(), edge))?"false":"true") <<
        ", \"is_output\":" << ((graph->outputEdges().end()==std::find(graph->outputEdges().begin(), graph->outputEdges().end(), edge))?"false":"true");


    //  now create "line" elements to represent the fan-in && fan-out of the edge
    const vector<canonical_ast::Node*> source_nodes = graph->upstreamNodes(edge);  // source
    const vector<canonical_ast::Node*> target_nodes = graph->downstreamNodes(edge); // target

    std::string delim0 = "\n\t,";
    os << ", \"sources\":[";
    if ( source_nodes.size() )
    {
        std::string source_delim = "";

        for ( size_t s = 0, S = source_nodes.size(); s < S; ++s )
        {
            os << source_delim << '"' << source_nodes[s]->id() << '"';
            source_delim = ", ";
        }
    }
    os << "], \"targets\":[";
    if ( target_nodes.size() )
    {
        std::string target_delim = "";
        for ( size_t t = 0, T = target_nodes.size(); t < T; ++t )
        {
            string target_node_id = target_nodes[t]->id();
            os << target_delim << '"' << target_nodes[t]->id() << '"';
            target_delim = ", ";
        }
    }
    os << "]}";
    return os;
}

ostream &canonical_ast::outputJson(canonical_ast::Graph *, canonical_ast::Node *node, ostream &os)
{
    os << " { \"class\":\"node\", \"id\":\"" << node->id() << "\" }";
    return os;
}


bool canonical_ast::serializeTo(WisdomContainerEntry *)
{
    return false;
}

bool canonical_ast::deserializeFrom(WisdomContainerEntry *)
{
    return false;
}

//  Canonical operation parameters
void canonical_ast::ConvolutionNode::captureNetworkParams(ConvolutionLayer* origNwLayer)
{
    Dims4 weightDims, biasDims;
    params().setBiasMode(origNwLayer->getBiasMode());
    params().setHasBiasTerm(origNwLayer->getBiasWeights().count > 0 ? true : false);
    params().setTopLeftPadding(origNwLayer->getTopLeftPadding());
    params().setBottomRightPadding(origNwLayer->getBottomRightPadding());
    params().setPaddingValue(origNwLayer->getPaddingValue());
    params().setStride(origNwLayer->getStride());
    params().setDilation(origNwLayer->getDilation());
    params().setWeights(origNwLayer->getKernelWeights());
    params().setBiasData(origNwLayer->getBiasWeights());
    params().setNumGroups(origNwLayer->getNumGroups());
    NvU32 kernChannels  = params().weights().count/
                         (origNwLayer->getNumOutputMaps() *
                          origNwLayer->getKernelSize().h *
                          origNwLayer->getKernelSize().w);
    weightDims.n = origNwLayer->getNumOutputMaps();
    weightDims.c = kernChannels;
    weightDims.h = origNwLayer->getKernelSize().h;
    weightDims.w = origNwLayer->getKernelSize().w;
    params().setWeightDims(weightDims);

    if (params().hasBiasTerm())
    {
        switch(origNwLayer->getBiasMode())
        {
            case BiasMode::bUNIFORM:
                biasDims.c = 1;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bCHANNEL:
                biasDims.c = params().biasData().count;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bm_ELEMENTWISE:
                biasDims.c = origNwLayer->getInput(0)->getDimensions().c;
                biasDims.h = origNwLayer->getInput(0)->getDimensions().h;
                biasDims.w = origNwLayer->getInput(0)->getDimensions().w;
                break;
            default:
                REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)origNwLayer->getBiasMode());
        }
        params().setBiasDims(biasDims);
    }

    return;
}

void canonical_ast::FullyConnectedNode::captureNetworkParams(FullyConnectedLayer* origNwLayer)
{
    Dims4 weightDims, biasDims;
    params().setBiasMode(origNwLayer->getBiasMode());
    params().setHasBiasTerm(origNwLayer->getBiasWeights().count > 0 ? true : false);
    params().setWeights(origNwLayer->getKernelWeights());
    params().setBiasData(origNwLayer->getBiasWeights());
    // the kernel weights of an inner product have the same dimensions as the input
    weightDims.n = origNwLayer->getNumOutputChannels();
    weightDims.c = origNwLayer->getInput(0)->getDimensions().c; // fixme: probably need fix
    weightDims.h = origNwLayer->getInput(0)->getDimensions().h; // fixme: probably need fix
    weightDims.w = origNwLayer->getInput(0)->getDimensions().w; // fixme: probably need fix
    params().setWeightDims(weightDims);

    if (params().hasBiasTerm())
    {
        switch(origNwLayer->getBiasMode())
        {
            case BiasMode::bUNIFORM:
                biasDims.c = 1;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bCHANNEL:
                biasDims.c = params().biasData().count;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bm_ELEMENTWISE:
                biasDims.c = origNwLayer->getInput(0)->getDimensions().c;
                biasDims.h = origNwLayer->getInput(0)->getDimensions().h;
                biasDims.w = origNwLayer->getInput(0)->getDimensions().w;
                break;
            default:
                REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)origNwLayer->getBiasMode());
        }
        params().setBiasDims(biasDims);
    }

    return;
}

void canonical_ast::ActivationNode::captureNetworkParams(ActivationLayer* origNwLayer)
{
    params().setActivationType(origNwLayer->getActivationType());
}

void canonical_ast::PoolingNode::captureNetworkParams(PoolingLayer* origNwLayer)
{
    params().setPoolType(origNwLayer->getPoolingType());
    params().setTopLeftPadding(origNwLayer->getTopLeftPadding());
    params().setBottomRightPadding(origNwLayer->getBottomRightPadding());
    params().setKernelDims(origNwLayer->getWindowSize());
    params().setStride(origNwLayer->getStride());
}

void canonical_ast::LRNNode::captureNetworkParams(LRNLayer* origNwLayer)
{
    params().setLocalSize(origNwLayer->getWindowSize());
    params().setAlpha(origNwLayer->getAlpha());
    params().setBeta(origNwLayer->getBeta());
    params().setK(origNwLayer->getK());
}

void canonical_ast::ScaleNode::captureNetworkParams(ScaleLayer* origNwLayer)
{
    Dims4 scaleDims, shiftDims, powerDims;
    params().setMode(origNwLayer->getMode());
    params().setShift(origNwLayer->getShift());
    params().setScale(origNwLayer->getScale());
    params().setPower(origNwLayer->getPower());
    params().setHasBiasTerm(origNwLayer->getShift().count > 0 ? true : false);
    switch(origNwLayer->getMode())
    {
        case ScaleMode::sUNIFORM:
            scaleDims.c = 1;
            scaleDims.h = 1;
            scaleDims.w = 1;
            break;
        case ScaleMode::sCHANNEL:
            scaleDims.c = params().scale().count;
            scaleDims.h = 1;
            scaleDims.w = 1;
            break;
        case ScaleMode::sm_ELEMENTWISE:
            scaleDims.c = origNwLayer->getInput(0)->getDimensions().c;
            scaleDims.h = origNwLayer->getInput(0)->getDimensions().h;
            scaleDims.w = origNwLayer->getInput(0)->getDimensions().w;
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown scale mode: %d", (int)origNwLayer->getMode());
    }
    params().setScaleDims(scaleDims);

    if (params().hasBiasTerm())
    {
        shiftDims = scaleDims;
        params().setShiftDims(shiftDims);
    }

    if (params().power().count > 0)
    {
        powerDims.c = 1;
        powerDims.h = 1;
        powerDims.w = 1;
        params().setPowerDims(powerDims);
    }
}

void canonical_ast::BatchNormNode::captureNetworkParams(BatchNormLayer* origNwLayer)
{
    Dims4 meanDims;
    Dims4 varianceDims;
    params().setMode(origNwLayer->getMode());
    params().setMean(origNwLayer->getMean());
    params().setVariance(origNwLayer->getVariance());
    params().setEpsilon(origNwLayer->getEpsilon());
    switch (origNwLayer->getParams().mode)
    {
        case BatchNormMode::bnUNIFORM:
            meanDims.c = 1;
            meanDims.h = 1;
            meanDims.w = 1;
            break;
        case BatchNormMode::bnm_CHANNEL:
            meanDims.c = params().mean().count;
            meanDims.h = 1;
            meanDims.w = 1;
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown batch norm mode: %d", (int)origNwLayer->getMode());
    }
    varianceDims = meanDims;
    params().setMeanDims(meanDims);
    params().setVarianceDims(varianceDims);
}

void canonical_ast::SoftMaxNode::captureNetworkParams(SoftMaxLayer* origNwLayer)
{
}

void canonical_ast::ConcatenationNode::captureNetworkParams(ConcatenationLayer* origNwLayer)
{
    params().setNumInputs(origNwLayer->getNumInputs());
}

void canonical_ast::SplitNode::captureNetworkParams(SliceLayer* origNwLayer)
{
    this->getParams().setNumOutputs(origNwLayer->getNumOutputs());
}

void canonical_ast::DeconvolutionNode::captureNetworkParams(DeconvolutionLayer* origNwLayer)
{
    Dims4 weightDims, biasDims;
    params().setBiasMode(origNwLayer->getBiasMode());
    params().setHasBiasTerm(origNwLayer->getBiasWeights().count > 0 ? true : false);
    params().setTopLeftPadding(origNwLayer->getTopLeftPadding());
    params().setBottomRightPadding(origNwLayer->getBottomRightPadding());
    params().setPaddingValue(origNwLayer->getPaddingValue());
    params().setStride(origNwLayer->getStride());
    params().setNumGroups(origNwLayer->getNumGroups());
    params().setDilation(origNwLayer->getDilation());
    params().setWeights(origNwLayer->getKernelWeights());
    params().setBiasData(origNwLayer->getBiasWeights());
    NvU32 kernChannels  = params().weights().count/
                         (origNwLayer->getNumOutputMaps() *
                          origNwLayer->getKernelSize().h *
                          origNwLayer->getKernelSize().w);

    weightDims.n = origNwLayer->getNumOutputMaps();
    weightDims.c = kernChannels;
    weightDims.h = origNwLayer->getKernelSize().h;
    weightDims.w = origNwLayer->getKernelSize().w;
    params().setWeightDims(weightDims);

    if (params().hasBiasTerm())
    {
        switch(origNwLayer->getBiasMode())
        {
            case BiasMode::bUNIFORM:
                biasDims.c = 1;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bCHANNEL:
                biasDims.c = params().biasData().count;
                biasDims.h = 1;
                biasDims.w = 1;
                break;
            case BiasMode::bm_ELEMENTWISE:
                biasDims.c = origNwLayer->getInput(0)->getDimensions().c;
                biasDims.h = origNwLayer->getInput(0)->getDimensions().h;
                biasDims.w = origNwLayer->getInput(0)->getDimensions().w;
                break;
            default:
                REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)origNwLayer->getBiasMode());
        }
        params().setBiasDims(biasDims);
    }

    return;
}

void canonical_ast::ElementWiseNode::captureNetworkParams(ElementWiseLayer* origNwLayer)
{
    params().setType(origNwLayer->getOperation());
}

// explicitly instantiate the priv maps of each node type
map<canonical_ast::Node*, canonical_ast::ConvolutionNode*> canonical_ast::NodeFactory::s_conv_priv =
    map<canonical_ast::Node*, canonical_ast::ConvolutionNode*>();

map<canonical_ast::Node*, canonical_ast::FullyConnectedNode*> canonical_ast::NodeFactory::s_fc_priv =
    map<canonical_ast::Node*, canonical_ast::FullyConnectedNode*>();

map<canonical_ast::Node*, canonical_ast::ActivationNode*> canonical_ast::NodeFactory::s_act_priv =
    map<canonical_ast::Node*, canonical_ast::ActivationNode*>();

map<canonical_ast::Node*, canonical_ast::PoolingNode*> canonical_ast::NodeFactory::s_pool_priv =
    map<canonical_ast::Node*, canonical_ast::PoolingNode*>();

map<canonical_ast::Node*, canonical_ast::LRNNode*> canonical_ast::NodeFactory::s_lrn_priv =
    map<canonical_ast::Node*, canonical_ast::LRNNode*>();

map<canonical_ast::Node*, canonical_ast::ScaleNode*> canonical_ast::NodeFactory::s_scale_priv =
    map<canonical_ast::Node*, canonical_ast::ScaleNode*>();

map<canonical_ast::Node*, canonical_ast::BatchNormNode*> canonical_ast::NodeFactory::s_bn_priv =
    map<canonical_ast::Node*, canonical_ast::BatchNormNode*>();

map<canonical_ast::Node*, canonical_ast::SoftMaxNode*> canonical_ast::NodeFactory::s_sm_priv =
    map<canonical_ast::Node*, canonical_ast::SoftMaxNode*>();

map<canonical_ast::Node*, canonical_ast::ConcatenationNode*> canonical_ast::NodeFactory::s_concat_priv =
    map<canonical_ast::Node*, canonical_ast::ConcatenationNode*>();

map<canonical_ast::Node*, canonical_ast::DeconvolutionNode*> canonical_ast::NodeFactory::s_deconv_priv =
    map<canonical_ast::Node*, canonical_ast::DeconvolutionNode*>();

map<canonical_ast::Node*, canonical_ast::ElementWiseNode*> canonical_ast::NodeFactory::s_ew_priv =
    map<canonical_ast::Node*, canonical_ast::ElementWiseNode*>();

map<canonical_ast::Node*, canonical_ast::SplitNode*> canonical_ast::NodeFactory::s_split_priv =
    map<canonical_ast::Node*, canonical_ast::SplitNode*>();

canonical_ast::ConvolutionNode* canonical_ast::NodeFactory::newConvNode(ConvolutionLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ConvolutionNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ConvolutionNode();
    d->captureNetworkParams(orig_nw_layer);

    s_conv_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::ConvolutionNode* canonical_ast::NodeFactory::newConvNodeFromJson(rapidjson::Value::ConstMemberIterator itr)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ConvolutionNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ConvolutionNode();
    // d->captureNetworkParams(orig_nw_layer);

    Dims4 weightDims, biasDims;

    // set padding 
    Dims2 TL(1,1), BR(1,1);
    vector<Dims2*> padding = {&TL, &BR};
    // canonical_ast::getDims2FromValue(itr->value.GetObject()["padding"][0], padding);
    d->params().setTopLeftPadding(TL);
    d->params().setBottomRightPadding(BR);
    d->params().setPaddingValue(0);

    // set stride
    Dims2 singleStride(1,1);
    // vector<Dims2*> Strides= {&singleStride};
    // canonical_ast::getDims2FromValue( itr->value.GetObject()["strides"][0], Strides);
    d->params().setStride(singleStride);
    
    d->params().setDilation(Dims2(1, 1));

    // weight loading
    NvS64 weightsize = itr->value.GetObject()["weights"].GetArray().Size();
    int offset=100000;
    vector<NvF32> m_weight(weightsize+offset*2);
    for(NvS64 i = 0; i < weightsize; i++){
        NvF32 value = itr->value.GetObject()["weights"][i].GetFloat();
        m_weight[i+offset] = value;
    }
    d->params().setWeights(Weights(DataType::FLOAT, (m_weight.data()+offset), weightsize));
    d->params().setNumGroups(1);

    weightDims.n = itr->value.GetObject()["weight_shape"][0].GetInt();
    weightDims.c = itr->value.GetObject()["weight_shape"][1].GetInt();
    weightDims.h = itr->value.GetObject()["weight_shape"][2].GetInt();
    weightDims.w = itr->value.GetObject()["weight_shape"][3].GetInt();
    d->params().setWeightDims(weightDims);

    // Bias
    if (itr->value.GetObject().HasMember("bias_info")){
        d->params().setHasBiasTerm(true);
        NvS64 biasSize = itr->value.GetObject()["bias_info"]["bias"].GetArray().Size();
        vector<NvF32> m_bias(biasSize+2*offset);
        for(NvS64 i = 0; i < biasSize; i++){
            NvF32 value = itr->value.GetObject()["bias_info"]["bias"][i].GetFloat();
            m_bias[i+offset] = value;
        }
        d->params().setBiasData(Weights(DataType::FLOAT, (m_bias.data()+offset), biasSize));
    
        switch (itr->value.GetObject()["bias_info"]["bias_shape"].GetArray().Size())
        {
        case 0:
        // TODO. not sure how bias shape of uniform bias looks like
            d->params().setBiasMode(BiasMode::bUNIFORM);
            biasDims.c = 1;
            biasDims.h = 1;
            biasDims.w = 1;
            break;
        case 1:
            d->params().setBiasMode(BiasMode::bCHANNEL);
            biasDims.c = d->params().biasData().count;
            biasDims.h = 1;
            biasDims.w = 1;
            break;
        case 3:
            d->params().setBiasMode(BiasMode::bm_ELEMENTWISE);
            biasDims.c = itr->value.GetObject()["bias_info"]["bias_shape"][0].GetInt();
            biasDims.h = itr->value.GetObject()["bias_info"]["bias_shape"][1].GetInt();
            biasDims.w = itr->value.GetObject()["bias_info"]["bias_shape"][2].GetInt();
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)d->params().biasMode());
        }
        d->params().setBiasDims(biasDims);
    }
    else{
        d->params().setHasBiasTerm(false);
    }

    s_conv_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::FullyConnectedNode* canonical_ast::NodeFactory::newFCNodeFromJson(rapidjson::Value::ConstMemberIterator itr)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::FullyConnectedNode* D;

    B b;
    D d;

    b = d = new canonical_ast::FullyConnectedNode();
    Dims4 weightDims, biasDims;

    // Weights
    NvS64 weightsize = itr->value.GetObject()["weights"].GetArray().Size();
    int offset=100000;
    vector<NvF32> m_weight(weightsize+2*offset);
    for(NvS64 i = 0; i < weightsize; i++){
        m_weight[i+offset]=itr->value.GetObject()["weights"][i].GetFloat();
    }
    d->params().setWeights(Weights(DataType::FLOAT, (m_weight.data()+offset), weightsize));
    switch(itr->value.GetObject()["weight_shape"].GetArray().Size()){
        case 2:
            weightDims.n = itr->value.GetObject()["weight_shape"][0].GetInt();
            weightDims.c = itr->value.GetObject()["weight_shape"][1].GetInt();
            weightDims.h = 1;
            weightDims.w = 1;
            break;
        case 4:
            weightDims.n = itr->value.GetObject()["weight_shape"][0].GetInt();
            weightDims.c = itr->value.GetObject()["weight_shape"][1].GetInt();
            weightDims.h = itr->value.GetObject()["weight_shape"][2].GetInt();
            weightDims.w = itr->value.GetObject()["weight_shape"][3].GetInt();
            break;
        default:
            gLogError<<"Unsupported weight shape for dense layer: "<< itr->name.GetString()<<endl;

    }
    d->params().setWeightDims(weightDims);

    // Bias
    if (itr->value.GetObject().HasMember("bias_info")){
        d->params().setHasBiasTerm(true);
        NvS64 biasSize = itr->value.GetObject()["bias_info"]["bias"].GetArray().Size();
        vector<NvF32> m_bias(biasSize+2*offset);
        for(NvS64 i = 0; i < biasSize; i++){
            NvF32 value = itr->value.GetObject()["bias_info"]["bias"][i].GetFloat();
            m_bias[i+offset] = value;
        }
        d->params().setBiasData(Weights(DataType::FLOAT, (m_bias.data()+offset), biasSize));
    
        switch (itr->value.GetObject()["bias_info"]["bias_shape"].GetArray().Size())
        {
        case 0:
        // TODO. not sure how bias shape of uniform bias looks like
            d->params().setBiasMode(BiasMode::bUNIFORM);
            biasDims.c = 1;
            biasDims.h = 1;
            biasDims.w = 1;
            break;
        case 1:
            d->params().setBiasMode(BiasMode::bCHANNEL);
            biasDims.c = d->params().biasData().count;
            biasDims.h = 1;
            biasDims.w = 1;
            break;
        case 3:
            d->params().setBiasMode(BiasMode::bm_ELEMENTWISE);
            biasDims.c = itr->value.GetObject()["bias_info"]["bias_shape"][0].GetInt();
            biasDims.h = itr->value.GetObject()["bias_info"]["bias_shape"][1].GetInt();
            biasDims.w = itr->value.GetObject()["bias_info"]["bias_shape"][2].GetInt();
            break;
        default:
            REPORT_ERROR(NvDlaError_BadParameter, "Unknown bias mode: %d", (int)d->params().biasMode());
        }
        d->params().setBiasDims(biasDims);
    }
    else{
        d->params().setHasBiasTerm(false);
    }

    s_fc_priv.insert(std::pair<B, D>(b, d));
    return d;

}
canonical_ast::FullyConnectedNode* canonical_ast::NodeFactory::newFCNode(FullyConnectedLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::FullyConnectedNode* D;

    B b;
    D d;

    b = d = new canonical_ast::FullyConnectedNode();
    d->captureNetworkParams(orig_nw_layer);

    s_fc_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::ActivationNode* canonical_ast::NodeFactory::newActivationNodeFromJson(rapidjson::Value::ConstMemberIterator itr)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ActivationNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ActivationNode();
    std::string name = itr->name.GetString();
    if(name.find("relu") != std::string::npos){
        d->params().setActivationType(ActivationType::kRELU);
    }
    else if (name.find("tanh") != std::string::npos){
        d->params().setActivationType(ActivationType::kTANH);   
    }
    else if (name.find("sigmoid") != std::string::npos){
        d->params().setActivationType(ActivationType::kSIGMOID);
    }
    s_act_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::ActivationNode* canonical_ast::NodeFactory::newActivationNode(ActivationLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ActivationNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ActivationNode();
    d->captureNetworkParams(orig_nw_layer);

    s_act_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::PoolingNode* canonical_ast::NodeFactory::newPoolingNodeFromJson(rapidjson::Value::ConstMemberIterator itr){
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::PoolingNode* D;

    B b;
    D d;

    b = d = new canonical_ast::PoolingNode();

    // Pooling Type
    std::string name(itr->name.GetString());
    // d->params().setPoolType(PoolingType::kMAX);
    if(name.find("max")!=std::string::npos){
        d->params().setPoolType(PoolingType::kMAX);
    }
    else if(name.find("min")!=std::string::npos){
        d->params().setPoolType(PoolingType::kMIN);
    }
    else if(name.find("avg")!=std::string::npos){
        d->params().setPoolType(PoolingType::kAVERAGE);
    }
    else{
        d->params().setPoolType(PoolingType::kMAX);
    }

    // Padding 
    //currently padding is [["0" , "0", "0", "0"]]
    Dims2 TL(0,0), BR(0,0);
    vector<Dims2*> padding = {&TL, &BR};
    canonical_ast::getDims2FromValue(itr->value.GetObject()["padding"][0], padding);
    d->params().setTopLeftPadding(TL);
    d->params().setBottomRightPadding(BR);

    // Kernel Dimensions
    Dims2 singleKernel(0,0);
    vector<Dims2*> Kernels= {&singleKernel};
    canonical_ast::getDims2FromValue( itr->value.GetObject()["pool_size"][0], Kernels);
    d->params().setKernelDims(singleKernel);

    //Stride 
    Dims2 singleStride(0,0);
    vector<Dims2*> Strides= {&singleStride};
    canonical_ast::getDims2FromValue( itr->value.GetObject()["strides"][0], Strides);
    d->params().setStride(singleStride);
    
    s_pool_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::PoolingNode* canonical_ast::NodeFactory::newPoolingNode(PoolingLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::PoolingNode* D;

    B b;
    D d;

    b = d = new canonical_ast::PoolingNode();
    d->captureNetworkParams(orig_nw_layer);

    s_pool_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::LRNNode* canonical_ast::NodeFactory::newLRNNode(LRNLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::LRNNode* D;

    B b;
    D d;

    b = d = new canonical_ast::LRNNode();
    d->captureNetworkParams(orig_nw_layer);

    s_lrn_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ScaleNode* canonical_ast::NodeFactory::newScaleNode(ScaleLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ScaleNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ScaleNode();
    d->captureNetworkParams(orig_nw_layer);

    s_scale_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::BatchNormNode* canonical_ast::NodeFactory::newBatchNormNode(BatchNormLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::BatchNormNode* D;

    B b;
    D d;

    b = d = new canonical_ast::BatchNormNode();
    d->captureNetworkParams(orig_nw_layer);

    s_bn_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::SoftMaxNode* canonical_ast::NodeFactory::newSoftMaxNodeFromJson(rapidjson::Value::ConstMemberIterator itr)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::SoftMaxNode* D;

    B b;
    D d;

    b = d = new canonical_ast::SoftMaxNode();
    s_sm_priv.insert(std::pair<B, D>(b, d));
    return d;
}
canonical_ast::SoftMaxNode* canonical_ast::NodeFactory::newSoftMaxNode(SoftMaxLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::SoftMaxNode* D;

    B b;
    D d;

    b = d = new canonical_ast::SoftMaxNode();
    d->captureNetworkParams(orig_nw_layer);

    s_sm_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ConcatenationNode* canonical_ast::NodeFactory::newConcatNode(ConcatenationLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ConcatenationNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ConcatenationNode();
    d->captureNetworkParams(orig_nw_layer);

    s_concat_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::SplitNode* canonical_ast::NodeFactory::newSplitNode(SliceLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::SplitNode* D;

    B b;
    D d;

    b = d = new canonical_ast::SplitNode();
    d->captureNetworkParams(orig_nw_layer);

    s_split_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::DeconvolutionNode* canonical_ast::NodeFactory::newDeconvNode(DeconvolutionLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::DeconvolutionNode* D;

    B b ;
    D d;

    b = d = new canonical_ast::DeconvolutionNode();
    d->captureNetworkParams(orig_nw_layer);

    s_deconv_priv.insert(std::pair<B, D>(b, d));
    return d;
}

canonical_ast::ElementWiseNode* canonical_ast::NodeFactory::newEWNode(ElementWiseLayer* orig_nw_layer)
{
    typedef typename canonical_ast::Node* B;
    typedef typename canonical_ast::ElementWiseNode* D;

    B b;
    D d;

    b = d = new canonical_ast::ElementWiseNode();
    d->captureNetworkParams(orig_nw_layer);

    s_ew_priv.insert(std::pair<B, D>(b, d));
    return d;
}

namespace canonical_ast
{

template <> canonical_ast::ConvolutionNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ConvolutionNode*>::iterator i = s_conv_priv.find(base);
    if ( i == s_conv_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::FullyConnectedNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::FullyConnectedNode*>::iterator i = s_fc_priv.find(base);
    if ( i == s_fc_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ActivationNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ActivationNode*>::iterator i = s_act_priv.find(base);
    if ( i == s_act_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::PoolingNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::PoolingNode*>::iterator i = s_pool_priv.find(base);
    if ( i == s_pool_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::LRNNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::LRNNode*>::iterator i = s_lrn_priv.find(base);
    if ( i == s_lrn_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ScaleNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ScaleNode*>::iterator i = s_scale_priv.find(base);
    if ( i == s_scale_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::BatchNormNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::BatchNormNode*>::iterator i = s_bn_priv.find(base);
    if ( i == s_bn_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::SoftMaxNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::SoftMaxNode*>::iterator i = s_sm_priv.find(base);
    if ( i == s_sm_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ConcatenationNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ConcatenationNode*>::iterator i = s_concat_priv.find(base);
    if ( i == s_concat_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::SplitNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::SplitNode*>::iterator i = s_split_priv.find(base);
    if ( i == s_split_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::DeconvolutionNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::DeconvolutionNode*>::iterator i = s_deconv_priv.find(base);
    if ( i == s_deconv_priv.end() )
        return NULL;
    return i->second;
}

template <> canonical_ast::ElementWiseNode* NodeFactory::nodeCast(canonical_ast::Node* base)
{
    map<canonical_ast::Node*, canonical_ast::ElementWiseNode*>::iterator i = s_ew_priv.find(base);
    if ( i == s_ew_priv.end() )
        return NULL;
    return i->second;
}


bool CanonicalParams::hasBiasTerm() const        { return false; }
void CanonicalParams::setHasBiasTerm(bool /*b*/) { }

}; // nvdla::priv::canonical_ast_interface

}; // nvdla::priv

}; // nvdla::
