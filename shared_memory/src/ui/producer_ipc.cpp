// Copyright (C) 2020 Adrian Böckenkamp
// This code is licensed under the BSD 3-Clause license (see LICENSE for details).

#include "producer_ipc.h"

#include <iostream>

#define log true
ProducerIPC::ProducerIPC(const QString &id, const QString &key_file_path)
  : AbstractIPC(id, key_file_path)
{
  if (log) {
    std::cout << (QString("PRODUCER Compiled with Qt v") + QT_VERSION_STR).toStdString().c_str();
    std::cout << "Creating shared memory with key=" << shared_memory.key().toStdString().c_str()
             << (file_key.isEmpty() ? " (hardcoded)" : " (loaded from file)");
  }
}

ProducerIPC::~ProducerIPC() { detach(); }

int ProducerIPC::begin(int desired_size, char **data)
{
  if (transaction_started) {
    if (log) {
      std::cout << "Already started an transaction, call end() first to start a new one.";
    }
    return -1;
  }
  if (!data) {
    if (log) {
      std::cout << "Invalid data pointer provided";
    }
    return -1;
  }

  if (shared_memory.isAttached()) {
    detach();
  }
  if (!shared_memory.create(int(desired_size))) {
    if (log) {
      std::cout << "Shared memory seems to be still existing, unable to create it. Trying to "
                  "recover by gaining ownership and detaching to delete it..."
                << shared_memory.errorString().toStdString().c_str() << "\n";
    }
    if(!shared_memory.attach()) {
      if (log) {
        std::cout << "Attach error.";
      }
    }
    if(!shared_memory.detach()) {
      if (log) {
        std::cout << "Detach error.";
      }
    }
    if (!shared_memory.create(int(desired_size))) { // we really still failed
      if (log) {
        std::cout << "Unable to create or recover shared memory segment: "
                 << shared_memory.errorString().toStdString().c_str() << "\n\nYou probably need to reboot to fix this.";
      }
      return -1;
    }
  }

  if (!sem_empty.acquire()) {
    if (log) {
      std::cout << "Unable to acquire system semaphore (sem_empty): " << sem_empty.errorString().toStdString().c_str();
    }
    return -1;
  }
  if (!shared_memory.lock()) {
    sem_full.release(); // undo acquire
    if (log) {
      std::cout << "Unable to lock shared memory: " << shared_memory.errorString().toStdString().c_str();
    }
    return -1;
  }
  *data = static_cast<char*>(shared_memory.data());
  transaction_started = true;
  return qMin(shared_memory.size(), desired_size);
}

void ProducerIPC::end()
{
  if (!transaction_started) {
    if (log) {
      std::cout << "You must call begin() first before calling end().";
    }
    return;
  }
  shared_memory.unlock();
  sem_full.release();
  transaction_started = false;
}
