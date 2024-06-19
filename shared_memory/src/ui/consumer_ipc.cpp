// Copyright (C) 2020 Adrian Böckenkamp
// This code is licensed under the BSD 3-Clause license (see LICENSE for details).

#include "consumer_ipc.h"

#include <iostream>
#include <QtConcurrent>
#include <QDebug>
#define log true
ConsumerIPC::ConsumerIPC(const QString &id, SharedMem *shared_memory_our, const QString &key_file_path)
  : AbstractIPC(id, key_file_path), data_acquired(false)
{
  this->shared_memory_our = shared_memory_our;
  std::cout<<"STTARRT\n";
  if (log) {
    std::cout << (QString("CONSUMER Compiled with Qt v") + QT_VERSION_STR).toStdString().c_str()<<"\n";
    std::cout << "Creating shared memory with key=" << shared_memory.key().toStdString().c_str()
             << (file_key.isEmpty() ? " (hardcoded)" : " (loaded from file)")<<"\n";
  }

  terminate = false;
  skip = false;
  std::cout<<"STTARRT\n";
  // future = QtConcurrent::run(this, &ConsumerIPC::updateThread);
}

ConsumerIPC::~ConsumerIPC()
{
  // Stop waiting in the thread and wait for its termination:
  if (log) {
    std::cout << "Requesting update thread to terminate..."<<"\n";
  }
  terminate = true;
  sem_full.release(); // fakes data to unblock the thread
  // future.waitForFinished();
  if (log) {
    std::cout << "Update thread has terminated successfully."<<"\n";
  }
}

void ConsumerIPC::updateThread()
{
  while (!terminate) {
    // Passively wait until a data is ready:
    if (log) {
      std::cout << "Update thread: Waiting for data..."<<"\n";
    }
    std::cout<<"GOT HERE!\n";
    if (!sem_full.acquire()) {
      if (log) {
        std::cout << "Unable to acquire system semaphore (sem_full): " << sem_full.errorString().toStdString().c_str()<<"\n";
      }
    }
    data_acquired = true;

    if (!terminate) {
      if (log) {
        std::cout << "Producer signaled that data is ready! Triggering UI thread..."<<"\n";
      }
      // Signal the GUI thread to display it:
      // emit available();
      shared_memory_our->loadFromMemory();

    }
  }
  if (log) {
    std::cout << "Update thread about to terminate..."<<"\n";
  }
}

int ConsumerIPC::begin(const char **data)
{
  if (!sem_full.acquire()) {
      if (log) {
        std::cout << "Unable to acquire system semaphore (sem_full): " << sem_full.errorString().toStdString().c_str()<<"\n";
      }
  }
  else {
    std::cout << "Exited acquire\n";
  }

  //   if (skip){
  //     std::cout << "Skipping";
  //     skip = false;
  //     return -1;
  //   }
  //   skip = true;
  // }


  // // Check to ensure that its allowed to call this since the thread was really signaled...this way,
  // // it could happen that sem_empty.release() is called in end() although sem_full.acquire() in
  // // updateThread() was never triggered:
  // if (!data_acquired) {
  //   if (log) {
  //     std::cout << "Data was not acquired yet. Wait until the signal ConsumerIPC::available() is "
  //               << "emitted and call this method in a slot upon being notified."<<"\n";
  //   }
  //   return -1;
  // }
  if (transaction_started) {
    if (log) {
      std::cout << "Already started an transaction, call end() first to start a new one."<<"\n";
    }
    return -1;
  }
  if (!data) { // invalid pointer?
    if (log) {
      std::cout << "Invalid pointer \'data\' provided."<<"\n";
    }
    return -1;
  }

  // Now, consume it:
  if (!shared_memory.attach()) {
    if (log) {
      std::cout << "Unable to attach to shared memory segment: " << shared_memory.errorString().toStdString().c_str()<<"\n";
    }
    sem_empty.release(); // undo (data may be lost)
    return -1;
  }

  if (!shared_memory.lock()) {
    if (log) {
      std::cout << "Unable to lock shared memory segment: " << shared_memory.errorString().toStdString().c_str()<<"\n";
    }
    shared_memory.detach(); // dito
    sem_empty.release();
    
    return -1;
  }
  *data = static_cast<const char*>(shared_memory.constData());
  transaction_started = true;
  return shared_memory.size();
}

void ConsumerIPC::end()
{
  if (!transaction_started) {
    if (log) {
      std::cout << "You must call begin() first before calling end()."<<"\n";
    }
    return;
  }
  data_acquired = false;
  if(!shared_memory.unlock()){
    if (log){
      std::cout<< "Unable to unlock shared memory segment: "<< shared_memory.errorString().toStdString().c_str()<<"\n";
    }
  }
  if(!shared_memory.detach()){
    if (log){
      std::cout<< "Unable to detach shared memory: " << shared_memory.errorString().toStdString().c_str()<<"\n";
    }
  }

  if (log) {
    std::cout << "Signaling that the produced data was consumed..."<<"\n";
  }
  if (!sem_empty.release()) {
    if (log) {
      std::cout << "Unable to release system semaphore (sem_empty): " << sem_empty.errorString().toStdString().c_str()<<"\n";
    }
  }
  transaction_started = false;
}
