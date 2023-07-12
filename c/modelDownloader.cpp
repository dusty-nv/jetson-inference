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
 
#include "modelDownloader.h"
#include "tensorNet.h"

#include "filesystem.h"
#include "logging.h"

#include <fstream>
#include <sstream>


nlohmann::json gModelManifest;
bool gManifestLoaded = false;


// LoadModelManifest
bool LoadModelManifest( nlohmann::json& models, const char* path )
{
	const std::string json_path = locateFile(path);
	
	if( json_path.length() == 0 )
	{
		LogError(LOG_TRT "failed to find model manifest file '%s'\n", path);
		return false;
	}
	
	try
	{
		std::ifstream file(json_path.c_str());
		file >> models;
	}
	catch (nlohmann::json::parse_error &e)
	{
		LogError(LOG_TRT "failed to load model manifest file '%s'\n", path);
		LogError("%s\n", e.what());
	}
	catch (...)
	{
		LogError(LOG_TRT "failed to load model manifest file '%s'\n", path);
		return false;
	}
	
	return true;
}


// LoadModelManifest
bool LoadModelManifest( const char* path )
{
	if( gManifestLoaded )
		return true;
	
	gManifestLoaded = LoadModelManifest(gModelManifest);
	return gManifestLoaded;
}


// FindModel
bool FindModel( const char* type, const char* name, nlohmann::json& models, nlohmann::json& model )
{
	for( auto it=models[type].begin(); it != models[type].end(); it++ )
	{
		nlohmann::json& model_json = it.value();

		if( strcasecmp(name, it.key().c_str()) == 0 )
		{
			model = model_json;
			return true;
		}

		nlohmann::json& alias = model_json["alias"];
		
		if( alias.empty() )
			continue;
		
		if( alias.is_string() )
		{
			const std::string alias_str = alias.get<std::string>();
			
			if( strcasecmp(name, alias_str.c_str()) == 0 )
			{
				model = model_json;
				return true;
			}
		}
		else if( alias.is_array() )
		{
			std::vector<std::string> aliases = model_json["alias"].get<std::vector<std::string>>();
		
			for( size_t n=0; n < aliases.size(); n++ )
			{
				if( strcasecmp(name, aliases[n].c_str()) == 0 )
				{
					model = model_json;
					return true;
				}
			}
		}
		
	}
	
	return false;
}


// FindModel
bool FindModel( const char* type, const char* name, nlohmann::json& model )
{
	if( !LoadModelManifest() )
		return false;
	
	return FindModel(type, name, gModelManifest, model);
}


// FindModel
bool FindModel( const char* type, const char* name )
{
	nlohmann::json model;
	return FindModel(type, name, model);
}


// DownloadModel
bool DownloadModel( nlohmann::json& model, uint32_t retries )
{
	const std::string dir = JSON_STR(model["dir"]);
	const std::string url = JSON_STR(model["url"]);
	const std::string tar = JSON_STR(model["tar"]);
	const std::string cmd = JSON_STR(model["cmd"]);
	
	if( locateFile("networks/" + dir).length() > 0 )
		return true;

	if( url.length() > 0 )
	{
		for( uint32_t retry=0; retry < retries; retry++ )
		{
			std::ostringstream cmd;
			
			cmd << "cd " << locateFile("networks") << " ; ";
			cmd << "wget ";
			
			if( retry > 0 )
				cmd << "--verbose ";
			else
				cmd << "--quiet ";
			
			cmd << "--show-progress --progress=bar:force:noscroll --no-check-certificate ";
			cmd << url;

			if( tar.length() > 0 )
			{
				cmd << " -O " << tar << " ; ";
				cmd << "tar -xzvf " << tar << " ; ";
				cmd << "rm " << tar;
			}
			else
			{
				cmd << " ; ";
			}
			
			const std::string cmd_str = cmd.str();
			
			LogVerbose(LOG_TRT "downloading model %s...\n", tar.c_str());
			LogVerbose("%s\n", cmd_str.c_str());
			
			const int result = system(cmd_str.c_str());
			
			if( result == 0 )
			{
				LogSuccess(LOG_TRT "downloaded model %s\n", tar.c_str());
				return true;
			}
			
			LogError(LOG_TRT "model download failed on retry %u with error %i\n", retry, result);
		}
	}
	
	if( cmd.length() > 0 )
	{
		LogVerbose(LOG_TRT "running model command:  %s\n", cmd.c_str());

		const int result = system(cmd.c_str());
			
		if( result == 0 )
		{
			LogSuccess(LOG_TRT "downloaded model to %s\n", dir.c_str());
			return true;
		}
	}
	
	LogError(LOG_TRT "failed to download model after %u retries\n", retries);
	LogInfo(LOG_TRT "if this error keeps occuring, see here for a mirror to download the models from:\n");
	LogInfo(LOG_TRT "   https://github.com/dusty-nv/jetson-inference/releases\n");
	
	return false;
}


// DownloadModel
bool DownloadModel( const char* type, const char* name, nlohmann::json& model, uint32_t retries )
{
	if( !FindModel(type, name, model) )
	{
		LogError(LOG_TRT "couldn't find built-in %s model '%s'\n", type, name);
		return false;
	}
	
	if( !DownloadModel(model, retries) )
	{
		LogError(LOG_TRT "failed to download built-in %s model '%s'\n", type, name);
		return false;
	}
	
	return true;
}
