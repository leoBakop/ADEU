/****************************************************************************
**
** Copyright (C) 2020 Adrian Böckenkamp
** Copyright (C) 2016 The Qt Company Ltd
** Contact: https://www.qt.io/licensing/
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "sharedMem.h"
#include "consumer_ipc.h"
#include "producer_ipc.h"

#include <QBuffer>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QThread>

#include <iostream>

// #ifdef SHARED_STRUCT
// #include <cppystruct.h>
// #endif

SharedMem::SharedMem()
  // #if consumer == 1
  //   aipc{new ConsumerIPC(UNIQUE_SHARED_MEMORY_NAME, this, SHARED_MEMORY_KEY_FILE)}
  // #else
  //   aipc{new ProducerIPC(UNIQUE_SHARED_MEMORY_NAME, SHARED_MEMORY_KEY_FILE)}
  // #endif
{

    aipc_cons = new ConsumerIPC(UNIQUE_SHARED_MEMORY_NAME1, this, SHARED_MEMORY_KEY_FILE);
    aipc_prod = new ProducerIPC(UNIQUE_SHARED_MEMORY_NAME2, SHARED_MEMORY_KEY_FILE);
// #if consumer == 1
//     std::cout << "I am the consumer.";  

//     // connect(&this->cipc, &ConsumerIPC::available, this, &ConsumerDialog::loadFromMemory);


// #else
//     std::cout << "I am the producer.";
  
//   // connect(ui.loadFromFileButton, &QPushButton::clicked, this, &ConsumerDialog::loadFromFile);
  
// #endif
}

SharedMem::~SharedMem()
{
    delete aipc_cons;
    delete aipc_prod;
    std::cout << "Deleted SharedMem object";
}

void SharedMem::loadToMemory() // producer logic (not active on default, see ctor)
{
    std::string readTerminal;
    std::cout<< "Enter message:";
    std::cin >> readTerminal;

    // Load into shared memory
    QBuffer buffer;
    buffer.open(QBuffer::ReadWrite);
    QDataStream out(&buffer);  
    out << QString(readTerminal.c_str());


    try {
    ScopedProducer sp(*reinterpret_cast<ProducerIPC*>(aipc_prod), int(buffer.size()));
    memcpy(sp.data(), buffer.data().data(), size_t(sp.size()));
    } catch (std::runtime_error &e) {
    std::cout<<"Error! scoped producer error\n";    
    }
    

}

void SharedMem::loadFromMemory() // consumer logic, active on default
{
    std::cout<<"Started loadFromMemory\n";
    QBuffer buffer;
    QDataStream ins(&buffer);

    QString final_msg;
  
    ScopedConsumer sc(*reinterpret_cast<ConsumerIPC*>(aipc_cons));
    if (sc.size() <= 0) {
      // std::cout<<"Unable to attach to shared memory segment.\nLoad an image first.";
      return;
    }
    // std::cout<< "sc size is: "<<sc.size() << "\n";
    buffer.setData(sc.data(), sc.size());
    buffer.open(QBuffer::ReadOnly);
    // std::cout<< "buffer size is: "<<buffer.size();
    
    ins >> final_msg;
    // std::cout<< "qstring size is: "<<final_msg.size() << "\n";
    std::cout<< "Message from producer: "<<final_msg.toStdString().c_str();
        
    return;
}
