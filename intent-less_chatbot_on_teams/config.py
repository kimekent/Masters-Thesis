#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

""" Bot Configuration """


class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "32abb787-97a7-4899-8f96-425ff22848da")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "woU8Q~vLCJipxLLWl1Lmr5Kxv8OcyN3aU.SOCadr")
