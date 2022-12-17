/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#ifndef __MODEL_DOWNLOADER_H__
#define __MODEL_DOWNLOADER_H__


#include "json.hpp"


/**
 * Download a pre-trained model given its JSON config.
 * @ingroup modelDownloader
 */
bool DownloadModel( nlohmann::json& model, uint32_t retries=2 );


/**
 * Download a pre-trained model and return its JSON config.
 * @ingroup modelDownloader
 */
bool DownloadModel( const char* type, const char* name, nlohmann::json& model, uint32_t retries=2 );


/**
 * Check if a model can be found in the model manifest.
 * @ingroup modelDownloader
 */
bool FindModel( const char* type, const char* name );


/**
 * Find a pre-trained model's config in the model manifest.
 * @ingroup modelDownloader
 */
bool FindModel( const char* type, const char* name, nlohmann::json& model );


/**
 * Find a pre-trained model's config in the model manifest.
 * @ingroup modelDownloader
 */
bool FindModel( const char* type, const char* name, nlohmann::json& models, nlohmann::json& model );


/**
 * Load the model manifest from a JSON file.
 * @ingroup modelDownloader
 */
bool LoadModelManifest( const char* path="networks/models.json" );


/**
 * Load the model manifest from a JSON file.
 * @ingroup modelDownloader
 */
bool LoadModelManifest( nlohmann::json& models, const char* path="networks/models.json" );


/**
 * Helper macro for getting a string from a JSON element
 * @ingroup modelDownloader
 */
#define JSON_STR(x)					(x.is_string() ? x.get<std::string>() : "")


/**
 * Helper macro for getting a string from a JSON element
 * @ingroup modelDownloader
 */
#define JSON_STR_DEFAULT(x, default)	(x.is_string() ? x.get<std::string>() : default)


#endif