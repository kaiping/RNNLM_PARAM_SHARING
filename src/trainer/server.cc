#include <list>
#include <tuple>
#include <queue>
#include "mshadow/tensor.h"
#include "trainer/server.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/cluster.h"

using namespace mshadow;
namespace singa {
Server::Server(int thread_id,int group_id, int server_id):
  thread_id_(thread_id),group_id_(group_id), server_id_(server_id){}

void Server::Setup(const UpdaterProto& proto,
    shared_ptr<ServerShard> shard, const vector<int>& slice2group){
	//VLOG(3) << "Parsing config file for host "<<hosts[id_] << " server id = " <<id_;
  updater_=shared_ptr<Updater>(Singleton<Factory<Updater>>::Instance()
      ->Create("Updater"));
  updater_->Init(proto);
  shard_=shard;
  slice2group_=slice2group;
}

void Server::Run(){
  LOG(INFO)<<"Server (group_id= "<<group_id_<<", id="<<server_id_<<") starts";
  dealer_=std::make_shared<Dealer>(2*thread_id_);
  dealer_->Connect(kInprocRouterEndpoint);
  auto cluster=Cluster::Get();
  Msg* ping=new Msg();
  ping->set_src(group_id_, server_id_, kServer);
  ping->set_dst(-1,-1,kStub);
  ping->add_frame("PING", 4);
  ping->set_type(kConnect);
  dealer_->Send(&ping);
  int syncEntry=0;
	//start recv loop and process requests
  while (true){
    Msg* msg=dealer_->Receive();
    if (msg==nullptr)
      break;
    Msg* response=nullptr, *sync=nullptr;
    int type=msg->type();
    if (type== kStop){
      msg->set_src(group_id_, server_id_, kServer);
      msg->set_dst(-1,-1, kStub);
      dealer_->Send(&msg);
      break;
    }else if (type==kConnect){
      // TODO remove receiving pong msg
      string pong((char*)msg->frame_data(), msg->frame_size());
      CHECK_STREQ("PONG", pong.c_str());
      DeleteMsg(&msg);
    }else if(type==kPut){
      response = HandlePut(&msg);
    }else{
      int pid=msg->trgt_second();
      if(shard_->find(pid)==shard_->end()){
        // delay the processing by re-queue the msg.
        response=msg;
        DLOG(ERROR)<<"Requeue msg";
    }else if(type==kSyncReminder){
      DeleteMsg(&msg);
      unsigned nchecks=0, nparams=shard_->size();
      while(nchecks<nparams
          &&group_locator_->at(shard_->at(syncEntry))!=group_id_){
        syncEntry=(syncEntry+1)%nparams;
        nchecks++;
      }
      if(nchecks==nparams) continue;
      auto param=shard_->at(syncEntry);
      if(param->local_version()!=param->version()){
        sync=param->GenSyncMsg(true);
        for(int i=0;i<cluster->nserver_groups();i++){
          if(i!=group_id_) {
            Msg* tmp=sync;
            if(i<cluster->nserver_groups()-1)
              tmp= new Msg(*sync);
            tmp->set_dst(i, server_locator_->at(param), kServer);
            tmp->set_src(group_id_, server_id_, kServer);
            dealer_->Send(&tmp);
            param->set_version(param->local_version());
            //DLOG(ERROR)<<"sync";
          }
        }
      }
    }else {
      int pid=msg->target_first();
      if(shard_->find(pid)==shard_->end()){
        // delay the processing by re-queue the msg.
        response=msg;
        LOG(ERROR)<<"Requeue";
>>>>>>> SINGA-8 Implement distributed Hogwild
      } else{
        auto param=shard_->at(pid);
        switch (type){
          case kGet:
            response=HandleGet(param, &msg);
            break;
          case kUpdate:
            response = HandleUpdate(param, &msg);
            break;
          case kSyncRequest:
            response = HandleSyncRequest(param, &msg);
            break;
          default:
            LOG(ERROR)<<"Unknown message type "<<type;
            break;
        }
      }
    }
    if (response!=nullptr)
      dealer_->Send(&response);
  }
  LOG(INFO)<<"Server (group_id= "<<group_id_<<", id="<<server_id_<<") stops";
}

Msg* Server::HandlePut(Msg **msg){
  int version=(*msg)->trgt_third();
  int pid=(*msg)->target_first();
  shared_ptr<Param> param=nullptr;
  if(shard_->find(pid)!=shard_->end()){
    LOG(ERROR)<<"Param ("<<pid<<") is put more than once";
    param=shard_->at(pid);
  }else{
    auto factory=Singleton<Factory<Param>>::Instance();
    param=shared_ptr<Param>(factory ->Create("Param"));
    param->set_id(pid);
    (*shard_)[pid]=param;
  }
  auto response=param->HandlePutMsg(msg);
  // must set version after HandlePutMsg which allocates the memory
  param->set_version(version);
  if(Cluster::Get()->nserver_groups()>1 &&
      group_locator_->at(param)!=group_id_){
    last_data_[pid]=std::make_shared<Blob<float>>();
    last_data_[pid]->ReshapeLike(param->data());
    last_data_[pid]->CopyFrom(param->data());
  }
  LOG(INFO)<<"Server put param "<<pid<<" size="<<param->size()<<" Bytes";
  return response;
}

Msg* Server::HandleGet(shared_ptr<Param> param, Msg **msg){
  if(param->version()<(*msg)->trgt_third())
    return *msg;
  else{
    auto reply= param->HandleGetMsg(msg);
    int paramid=reply->trgt_first(), slice=reply->trgt_second();
    reply->set_trgt(paramid, slice, param->version());
  }
}

Msg* Server::HandleUpdate(shared_ptr<Param> param, Msg **msg) {
  auto* tmp=static_cast<Msg*>((*msg)->CopyAddr());
  tmp->SwapAddr();
  int paramid=(*msg)->trgt_first();
  int sliceid=(*msg)->trgt_second();
  int step=(*msg)->trgt_third();
  bool copy=param->ParseUpdateMsg(msg);
  updater_->Update(step, param);
  param->set_version(param->version()+1);
  auto response=param->GenUpdateResponseMsg(copy);
  response->set_trgt(paramid, sliceid, param->version());
  response->SetAddr(tmp);
  delete tmp;
  return response;
}

Msg* Server::HandleSyncRequest(shared_ptr<Param> param, Msg **msg){
  Msg* response=nullptr;
  auto shape=Shape1(param->size());
  CHECK_EQ((*msg)->frame_size(), param->size()*sizeof(float));
  Tensor<cpu, 1> tmp(static_cast<float*>((*msg)->frame_data()), shape);
  Tensor<cpu, 1> cur(param->mutable_cpu_data(), shape);
  if(group_locator_->at(param)==group_id_){
    cur+=tmp;
    param->set_local_version(param->local_version()+1);
  }else{
    TensorContainer<cpu, 1> diff(shape);
    Tensor<cpu, 1> prev(last_data_[param->id()]->mutable_cpu_data(), shape);
    diff=cur-prev;
    (*msg)->next_frame();
    int bandwidth;
    sscanf(static_cast<char*>((*msg)->frame_data()), "%d", &bandwidth);
    if(bandwidth>0){
      response=new Msg();
      response->set_type(kSyncRequest);
      response->set_target(param->id(), param->version());
      response->add_frame(diff.dptr, param->size()*sizeof(float));
      (*msg)->SwapAddr();
      response->SetAddr(*msg);
      prev=diff+tmp;
      Copy(cur, prev);
    }else{
      Copy(prev, tmp);
      cur=tmp+diff;
    }
  }
  DeleteMsg(msg);
  return response;
}
} /* singa */
