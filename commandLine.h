/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __COMMAND_LINE_H_
#define __COMMAND_LINE_H_


/**
 * commandLine parser class
 * @ingroup util
 */
class commandLine
{
public:
	/**
	 * constructor
	 */
	commandLine( const int argc, char** argv );


	/**
	 * Checks to see whether the specified flag was included on the 
	 * command line.   For example, if argv contained "--foo", then 
	 * GetFlag("foo") would return true.
	 *
	 * @returns true, if the flag with argName was found
	 *          false, if the flag with argName was not found
	 */
	bool GetFlag( const char* argName );

	
	/**
	 * Get float argument.  For example if argv contained "--foo=3.14159", 
	 * then GetInt("foo") would return 3.14159.0f
	 *
	 * @returns 0, if the argumentcould not be found.
	 *          Otherwise, returns the value of the argument.
	 */
	float GetFloat( const char* argName );


	/**
	 * Get integer argument.  For example if argv contained "--foo=100", 
	 * then GetInt("foo") would return 100.
	 *
	 * @returns 0, if the argument could not be found.
	 *          Otherwise, returns the value of the argument. 
	 */
	int GetInt( const char* argName );


	/**
	 * Get string argument.  For example if argv contained "--foo=bar",
	 * then GetString("foo") would return "bar".
	 *
	 * @returns NULL, if the argument could not be found.
	 *          Otherwise, returns a pointer to the argument value string
	 *          from the argv array.
	 */
	const char* GetString( const char* argName );


protected:

	int argc;
	char** argv;
};



#endif

