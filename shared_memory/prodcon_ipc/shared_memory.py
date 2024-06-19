#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################
##
## Copyright (C) 2020 Adrian Böckenkamp
## Copyright (C) 2013 Riverbank Computing Limited
## Copyright (C) 2010 Hans-Peter Jansen <hpj@urpla.net>.
## Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
## All rights reserved.
##
## This file is part of the examples of PyQt.
##
## $QT_BEGIN_LICENSE:LGPL$
## Commercial Usage
## Licensees holding valid Qt Commercial licenses may use this file in
## accordance with the Qt Commercial License Agreement provided with the
## Software or, alternatively, in accordance with the terms contained in
## a written agreement between you and Nokia.
##
## GNU Lesser General Public License Usage
## Alternatively, this file may be used under the terms of the GNU Lesser
## General Public License version 2.1 as published by the Free Software
## Foundation and appearing in the file LICENSE.LGPL included in the
## packaging of this file.  Please review the following information to
## ensure the GNU Lesser General Public License version 2.1 requirements
## will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
##
## In addition, as a special exception, Nokia gives you certain additional
## rights.  These rights are described in the Nokia Qt LGPL Exception
## version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
##
## GNU General Public License Usage
## Alternatively, this file may be used under the terms of the GNU
## General Public License version 3.0 as published by the Free Software
## Foundation and appearing in the file LICENSE.GPL included in the
## packaging of this file.  Please review the following information to
## ensure the GNU General Public License version 3.0 requirements will be
## met: http://www.gnu.org/copyleft/gpl.html.
##
## If you have questions regarding the use of this file, please contact
## Nokia at qt-info@nokia.com.
## $QT_END_LICENSE$
##
#############################################################################

from __future__ import with_statement

# Abstract: this Python app contains the producer logic. Upon selecting an image file to load, it
#           will be displayed in the UI and copied to the shared memory (producing). The consumer
#           can then display (= consume) it as well. Don't (try to) load another image if the
#           previous image hasn't been consumed; otherwise the app will block (until consumed). By
#           changing CONSUMER of both apps, the behavior (consumer/producer) can be inverted.

CONSUMER = 0

UNIQUE_SHARED_MEMORY_NAME1 = "MySharedMemoryDefault11111"
UNIQUE_SHARED_MEMORY_NAME2 = "MySharedMemoryDefault22222"
SHARED_MEMORY_KEY_FILE1 = "shared_memory1.key"
SHARED_MEMORY_KEY_FILE2 = "shared_memory2.key"

from PyQt5.QtCore import QBuffer, QDataStream, QSharedMemory
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
import sys
import os
cur_dic = os.path.dirname(__file__)
print("current dic: ", cur_dic)
sys.path.append(cur_dic)
import signal
import os.path
import subprocess
from producer_ipc import ProducerIPC
from consumer_ipc import ConsumerIPC


class SharedMem:
    def __init__(self):
        self.producer_ipc = ProducerIPC(UNIQUE_SHARED_MEMORY_NAME1, SHARED_MEMORY_KEY_FILE1)
        self.consumer_ipc = ConsumerIPC(UNIQUE_SHARED_MEMORY_NAME2, SHARED_MEMORY_KEY_FILE2)


    def load_to_memory(self, message):  # producer slot


        # Get the message data:
        buf = QBuffer()
        buf.open(QBuffer.ReadWrite)
        out = QDataStream(buf)
        # out << str(message)
        out.writeQString(str(message))

        try:
            from producer_ipc import ScopedProducer
            with ScopedProducer(self.producer_ipc, buf.size()) as sp:
                # Copy data from buf into shared memory area:
                sp.data()[:sp.size()] = buf.data().data()[:sp.size()]
        except Exception as err:
            print(str(err))

        # print("Producer message sent")

    def load_from_memory(self):  # consumer slot
        buf = QBuffer()
        ins = QDataStream(buf)
        message = 0

        try:
            from consumer_ipc import ScopedConsumer
            with ScopedConsumer(self.consumer_ipc) as sc:
                buf.setData(sc.data())
                buf.open(QBuffer.ReadOnly)
                # ins >> message
                message = ins.readQString()
        except Exception as err:
            print(str(err))

        # print("Consumer received message: ", message)
        return message

# if __name__ == '__main__':

#     shared = SharedMem()
#     cpp_args = ['gnome-terminal', '--','/home/iason/shared_memory/shared-memory-pycpp-example-master_tweaked/shared_memory_cpp', '1']
    
#     cpp_proc = subprocess.Popen(args=cpp_args, preexec_fn=os.setpgrp)
#     print("proc pid: ", cpp_proc.pid)
#     # os.startfile(cpp_args)
#     try:
#         while True:
#             if CONSUMER == 0:
#                 shared.load_to_memory()
#             elif CONSUMER == 1:
#                 shared.load_from_memory()
#     except KeyboardInterrupt:
#         print("terminating cpp_proc")
#         # cpp_proc.kill()
#         os.killpg(cpp_proc.pid, signal.SIGINT)
#         print("terminated")
