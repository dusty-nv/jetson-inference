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
var connections = {};

function reportError(msg) {
  console.log(msg);
}
 
function getWebsocketProtocol() {
  return window.location.protocol == 'https:' ? 'wss://' : 'ws://';
}

function getWebsocketURL(name, port=8554) {  // wss://192.168.1.2:8554/name
  return `${getWebsocketProtocol()}${window.location.hostname}:${port}/${name}`;
}
  
function checkMediaDevices() {
  return (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !navigator.mediaDevices.enumerateDevices) ? false : true;
}

function hasConnectionType(type) {  // 'inbound' or 'outbound'
	for( const connection in connections )
		if( connections[connection].type == type )
			return true;
	return false;
}

function onIncomingSDP(url, sdp) {
  console.log('Incoming SDP: (%s)' + JSON.stringify(sdp), url);

  function onLocalDescription(desc) {
    console.log('Local description (%s)\n' + JSON.stringify(desc), url);
    connections[url].webrtcPeer.setLocalDescription(desc).then(function() {
      connections[url].websocket.send(JSON.stringify({ type: 'sdp', 'data': connections[url].webrtcPeer.localDescription }));
    }).catch(reportError);
  }

  connections[url].webrtcPeer.setRemoteDescription(sdp).catch(reportError);

  if( connections[url].type == 'inbound' ) {
    connections[url].webrtcPeer.createAnswer().then(onLocalDescription).catch(reportError);
  }
  else if( connections[url].type == 'outbound' ) {
		const constraints = {
			audio: false,
			video: { 
				/*width: { max: 320 },
				height: { max: 240 },*/
				frameRate: { max: 15 },
				deviceId: connections[url].deviceId,
			}
		};
					
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
      console.log('Adding local stream (deviceId=%s)', connections[url].deviceId);
      connections[url].webrtcPeer.addStream(stream);
      connections[url].webrtcPeer.createAnswer().then(onLocalDescription).catch(reportError);
    }).catch(reportError);
  }
}

function onIncomingICE(url, ice) {
  var candidate = new RTCIceCandidate(ice);
  console.log('Incoming ICE (%s)\n' + JSON.stringify(ice), url);
  connections[url].webrtcPeer.addIceCandidate(candidate).catch(reportError);
}

function onAddRemoteStream(event) {
  var url = event.srcElement.url;
  console.log('Adding remote stream to HTML video player (%s)', url);
  connections[url].videoElement.srcObject = event.streams[0];
  connections[url].videoElement.play();
}

function onIceCandidate(event) {
  var url = event.srcElement.url;

  if (event.candidate == null)
    return;

  console.log('Sending ICE candidate out (%s)\n' + JSON.stringify(event.candidate), url);
  connections[url].websocket.send(JSON.stringify({'type': 'ice', 'data': event.candidate }));
}

function getConnectionStats(url, reportType) { 
  if( reportType == undefined ) 
    reportType = 'all'; 

  connections[url].webrtcPeer.getStats(null).then((stats) => { 
    let statsOutput = ''; 

    stats.forEach((report) => { 
      if( reportType == 'inbound-rtp' && report.type === 'inbound-rtp' && report.kind === 'video') { 
        statsOutput += `# inbound-rtp\n`; 

        if( connections[url].bytesReceived != undefined ) 
          statsOutput += `bitrate:          ${((report.bytesReceived - connections[url].bytesReceived) / 125000).toFixed(3)} mbps\n`; 

        connections[url].bytesReceived = report.bytesReceived; 

        statsOutput += `bytesReceived:    ${report.bytesReceived}\n`; 
        statsOutput += `packetsReceived:  ${report.packetsReceived}\n`; 
        statsOutput += `packetsLost:      ${report.packetsLost}\n`; 
        statsOutput += `framesReceived:   ${report.framesReceived}\n`; 
        statsOutput += `framesDropped:    ${report.framesDropped}\n`; 
        statsOutput += `frameWidth:       ${report.frameWidth}\n`; 
        statsOutput += `frameHeight:      ${report.frameHeight}\n`; 
        statsOutput += `framesPerSecond:  ${report.framesPerSecond}\n`; 
        statsOutput += `keyFramesDecoded: ${report.keyFramesDecoded}\n`; 
        statsOutput += `jitter:           ${report.jitter}\n`; 
      } 
      else if( reportType =='outbound-rtp' && report.type === 'outbound-rtp' && report.kind === 'video') { 
        statsOutput += `# outbound-rtp\n`; 

        if( connections[url].bytesSent != undefined ) 
          statsOutput += `bitrate:          ${((report.bytesSent - connections[url].bytesSent) / 125000).toFixed(3)} mbps\n`; 

        connections[url].bytesSent = report.bytesSent; 

        statsOutput += `bytesSent:        ${report.bytesSent}\n`; 
        statsOutput += `packetsSent:      ${report.packetsSent}\n`; 
        statsOutput += `packetsResent:    ${report.retransmittedPacketsSent}\n`; 
        statsOutput += `framesSent:       ${report.framesSent}\n`; 
        statsOutput += `frameWidth:       ${report.frameWidth}\n`; 
        statsOutput += `frameHeight:      ${report.frameHeight}\n`; 
        statsOutput += `framesPerSecond:  ${report.framesPerSecond}\n`; 
        statsOutput += `keyFramesSent:    ${report.keyFramesEncoded}\n`; 
      } 
      else if( reportType == 'all' || reportType == report.type ) { 
        statsOutput += `<h2>Report: ${report.type}</h2>\n<strong>ID:</strong> ${report.id}<br>\n` + 
        `<strong>Timestamp:</strong> ${report.timestamp}\n`; 

        Object.keys(report).forEach((statName) => { 
          if (statName !== 'id' && statName !== 'timestamp' && statName !== 'type') 
            statsOutput += `<strong>${statName}:</strong> ${report[statName]}\n`; 
        }); 
      } 
    }); 

    var statsElement = (connections[url].type == 'inbound') ? 'connection-stats-play' : 'connection-stats-send'; 
    document.getElementById(statsElement).innerHTML = statsOutput; 
  }); 
} 

function onServerMessage(event) {
  var msg;
  var url = event.srcElement.url;

  try {
    msg = JSON.parse(event.data);
  } catch (e) {
    return;
  }

  if( !connections[url].webrtcPeer ) {
    connections[url].webrtcPeer = new RTCPeerConnection(connections[url].webrtcConfig);
    connections[url].webrtcPeer.url = url;

    connections[url].webrtcPeer.onconnectionstatechange = (ev) => {
      console.log('WebRTC connection state (%s) ' + connections[url].webrtcPeer.connectionState, url);
      
      if( connections[url].webrtcPeer.connectionState == 'connected' )
        setInterval(getConnectionStats, 1000, url, connections[url].type == 'inbound' ? 'inbound-rtp' : 'outbound-rtp');
    }

    if( connections[url].type == 'inbound' ) {
      connections[url].webrtcPeer.ontrack = onAddRemoteStream;
    }
    
    connections[url].webrtcPeer.onicecandidate = onIceCandidate;
  }

  switch (msg.type) {
    case 'sdp': onIncomingSDP(url, msg.data); break;
    case 'ice': onIncomingICE(url, msg.data); break;
    default: break;
  }
}

function playStream(url, videoElement) {
  console.log('playing stream ' + url); 
   
  connections[url] = {};

  connections[url].type = 'inbound';
  connections[url].videoElement = videoElement;
  connections[url].webrtcConfig = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] };
   
  connections[url].websocket = new WebSocket(url);
  connections[url].websocket.addEventListener('message', onServerMessage);
}

function sendStream(url, deviceId) {
  console.log(`sending stream ${url}  (deviceId=${deviceId})`);

	if( url in connections && connections[url].type == 'outbound' ) {
		// replace the outbound stream in the existing connection
		replaceStream(url, deviceId);
		return false;
	}
	else {
		// create a new outbound connection
		connections[url] = {};

		connections[url].type = 'outbound';
		connections[url].deviceId = deviceId;
		connections[url].webrtcConfig = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] };

		connections[url].websocket = new WebSocket(url);
		connections[url].websocket.addEventListener('message', onServerMessage);
		
		return true;
	}
}

function replaceStream(url, deviceId) {
	console.log(`replacing stream for outbound WebRTC connection to ${url}`);
	console.log(`old device ID:  ${connections[url].deviceId}`);
	console.log(`new device ID:  ${deviceId}`);
	
	var constraints = {'audio': false, 'video': { deviceId: deviceId }};
	
	navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
		const [videoTrack] = stream.getVideoTracks();
		const sender = connections[url].webrtcPeer.getSenders().find((s) => s.track.kind === videoTrack.kind);
		console.log('found sender:', sender);
		sender.replaceTrack(videoTrack);
		connections[url].deviceId = deviceId;
	}).catch(reportError);
}
