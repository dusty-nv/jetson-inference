/*
 * clientside multi-stream WebRTC connection handler
 */
var connections = {};

function reportError(errmsg) {
	console.error(errmsg);
}

function onIncomingSDP(url, sdp) {
	console.log("Incoming SDP (%s)\n" + JSON.stringify(sdp), url);
	
	function onLocalDescription(desc) {
		console.log("Local description (%s)\n" + JSON.stringify(desc), url);
		connections[url].webrtcPeer.setLocalDescription(desc).then(function() {
			connections[url].websocket.send(JSON.stringify({ type: "sdp", "data": connections[url].webrtcPeer.localDescription }));
		}).catch(reportError);
	}

	connections[url].webrtcPeer.setRemoteDescription(sdp).catch(reportError);
	connections[url].webrtcPeer.createAnswer().then(onLocalDescription).catch(reportError);
}

function onIncomingICE(url, ice) {
	var candidate = new RTCIceCandidate(ice);
	console.log("Incoming ICE (%s)\n" + JSON.stringify(ice), url);
	connections[url].webrtcPeer.addIceCandidate(candidate).catch(reportError);
}

function onAddRemoteStream(event) {
	var url = event.srcElement.url;
	console.log("Adding remote stream to HTML video player (%s)", url);
	connections[url].stream = event.streams[0];
	connections[url].videoElement.srcObject = event.streams[0];
	connections[url].videoElement.play();
}

function onIceCandidate(event) {
	var url = event.srcElement.url;
	
	if( event.candidate == null )
		return;

	console.log("Sending ICE candidate out (%s)\n" + JSON.stringify(event.candidate), url);
	connections[url].websocket.send(JSON.stringify({ "type": "ice", "data": event.candidate }));
}


function onServerMessage(event) {
	//console.log(event);
	//console.log(connections);
	
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
		
		connections[url].webrtcPeer.onconnectionstatechange = (ev) => { console.log("WebRTC connection state:  " + connections[url].webrtcPeer.connectionState); }
		connections[url].webrtcPeer.ontrack = onAddRemoteStream;
		connections[url].webrtcPeer.onicecandidate = onIceCandidate;
	}

	switch (msg.type) {
		case "sdp": onIncomingSDP(url, msg.data); break;
		case "ice": onIncomingICE(url, msg.data); break;
		default: break;
	}
}

var wait = (ms) => {
	const start = Date.now();
	let now = start;
	while (now - start < ms) {
		now = Date.now();
	}
}

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.webrtc = 
{
    // this function gets called from the dash app
    playStream: function(streamConfig) 
    {
		console.log("window.dash_clientside.webrtc.playStream() => \n" + streamConfig)
		console.log("window location:  " + window.location);
		
	     streamConfig = JSON.parse(streamConfig)

		// check for HTTPS/SSL
		var ws_protocol = 'ws://';

		if( 'sslCert' in streamConfig['output'] )
			ws_protocol = 'wss://';
		
		// get the WebRTC websocket URL
		var url = ws_protocol + window.location.hostname + ':' + streamConfig['output']['resource']['port'] + streamConfig['output']['resource']['path'];
		console.log("websocket URL:  " + url);
		
		if( !connections[url] ) 
		{
			// create a new connection entry
			connections[url] = {};

			connections[url].videoElement = document.getElementById(streamConfig['video_player']);
			connections[url].webrtcConfig = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] }; // TODO pull stun_server from config
			connections[url].streamConfig = streamConfig;
			
			console.log(connections);
			
			// open websocket to the server
			connections[url].websocket = new WebSocket(url)
			connections[url].websocket.addEventListener('message', onServerMessage);
		}
		else 
		{
			// reconnect the video player to the stream
			console.warn("connection already opened to %s", url);
			
			connections[url].videoElement = document.getElementById(streamConfig['video_player']);
			connections[url].videoElement.srcObject = connections[url].stream;
		}
    }
}

/*
window.onload = function() {
	var vidstream = document.getElementById("stream");
	var config = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] };
	playStream(vidstream, null, null, null, config, function (errmsg) { console.error(errmsg); });
};*/