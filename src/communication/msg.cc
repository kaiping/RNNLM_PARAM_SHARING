#include "communication/msg.h"

namespace singa {

#ifdef USE_ZMQ
Msg::Msg() {
  msg_ = zmsg_new();
}

Msg::~Msg() {
  if (msg_ != nullptr)
    zmsg_destroy(&msg_);
}

void Msg::add_frame(const void* addr, int nBytes) {
  zmsg_addmem(msg_, addr, nBytes);
}

int Msg::frame_size() {
  return zframe_size(frame_);
}

void* Msg::frame_data() {
  return zframe_data(frame_);
}

bool Msg::next_frame() {
  frame_ = zmsg_next(msg_);
  return frame_ != nullptr;
}

void Msg::ParseFromZmsg(zmsg_t* msg) {
  char* tmp = zmsg_popstr(msg);
  sscanf(tmp, "%d %d %d %d %d %d",
         &src_, &dst_, &type_, &trgt_first_, &trgt_second_, &trgt_third_);
  frame_ = zmsg_next(msg);
  msg_ = msg;
}

zmsg_t* Msg::DumpToZmsg() {
  zmsg_pushstrf(msg_, "%d %d %d %d %d %d",
      src_, dst_, type_, trgt_first_, trgt_second_, trgt_third_);
  zmsg_t *tmp = msg_;
  msg_ = nullptr;
  return tmp;
}
#endif

}  // namespace singa

