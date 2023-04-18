/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

function rest_get(url, quiet=false) {
  return fetch(url).then(function(response) {
    return response.json();
  }).then(function(json) {
    if( !quiet ) {
			console.log(`GET response from ${url}`);
			console.log(json); 
		}
    return json;
  });
}

function rest_put(url, data, response_handler=null, quiet=false) {
  fetch(url, {
    method: 'PUT',
    body: JSON.stringify(data),
    headers: {'Content-type': 'application/json; charset=UTF-8'}
  }).then(function(response) {
		if( !quiet ) {
			console.log(`PUT response from ${url}`);
			console.log(response);
		}
		if( response_handler != undefined )
			response_handler(response);
  });  
}

function rest_post(url, data, response_handler=null, quiet=false) {
  return fetch(url, {
    method: 'POST',
    body: JSON.stringify(data),
    headers: {'Content-type': 'application/json; charset=UTF-8'}
  }).then(function(response) {
    return response.json();
  }).then(function(json) {
		if( !quiet ) {
			console.log(`POST response from ${url}`);
			console.log(json);
		}
		if( response_handler != undefined )
			response_handler(json);
    return json;
  });  
}