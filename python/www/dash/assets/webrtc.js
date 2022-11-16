var html5VideoElement;
var websocketConnection;
var webrtcPeerConnection;
var webrtcConfiguration;
var reportError;


function onLocalDescription(desc) {
	console.log("Local description: " + JSON.stringify(desc));
	webrtcPeerConnection.setLocalDescription(desc).then(function() {
	websocketConnection.send(JSON.stringify({ type: "sdp", "data": webrtcPeerConnection.localDescription }));
	}).catch(reportError);
}


function onIncomingSDP(sdp) {
	console.log("Incoming SDP: " + JSON.stringify(sdp));
	webrtcPeerConnection.setRemoteDescription(sdp).catch(reportError);
	webrtcPeerConnection.createAnswer().then(onLocalDescription).catch(reportError);
}


function onIncomingICE(ice) {
	var candidate = new RTCIceCandidate(ice);
	console.log("Incoming ICE: " + JSON.stringify(ice));
	webrtcPeerConnection.addIceCandidate(candidate).catch(reportError);
}


function onAddRemoteStream(event) {
	console.log("Adding remote stream to HTML video player");
	html5VideoElement.srcObject = event.streams[0];
}


function onIceCandidate(event) {
	if (event.candidate == null)
		return;

	console.log("Sending ICE candidate out: " + JSON.stringify(event.candidate));
	websocketConnection.send(JSON.stringify({ "type": "ice", "data": event.candidate }));
}


function onServerMessage(event) {
	var msg;

	try {
		msg = JSON.parse(event.data);
	} catch (e) {
		return;
	}

	if (!webrtcPeerConnection) {
		webrtcPeerConnection = new RTCPeerConnection(webrtcConfiguration);
		webrtcPeerConnection.ontrack = onAddRemoteStream;
		webrtcPeerConnection.onicecandidate = onIceCandidate;
	}

	switch (msg.type) {
		case "sdp": onIncomingSDP(msg.data); break;
		case "ice": onIncomingICE(msg.data); break;
		default: break;
	}
}

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

var wait = (ms) => {
	const start = Date.now();
	let now = start;
	while (now - start < ms) {
		now = Date.now();
	}
}

window.dash_clientside.webrtc = {
    // this function gets called from the dash app
    playStream: function(stream_config) {
		console.log("window.dash_clientside.webrtc.playStream() => " + stream_config)
	     stream_config = JSON.parse(stream_config)
		
		html5VideoElement = document.getElementById("video_player");					 // TODO dynamically get name of video player
		webrtcConfiguration = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] }; // TODO pull stun_server from config
		reportError = function (errmsg) { console.error(errmsg); };
		
		console.log("video player: " + html5VideoElement);
		console.log("window location:  " + window.location);

		// check for HTTPS/SSL
		var ws_protocol = "ws://";

		if( "sslCert" in stream_config["output"] )
			ws_protocol = "wss:///";
		
		// get the WebRTC websocket URL
		var ws_url = ws_protocol + window.location.hostname + ":" + stream_config["output"]["resource"]["port"] + stream_config["output"]["resource"]["path"];
		console.log("websocket URL:  " + ws_url);
		
		// open websocket to the server
		websocketConnection = new WebSocket(ws_url)
		websocketConnection.addEventListener("message", onServerMessage);
		
		//wait(2500);
		//console.log("returning from playStream()");
    }
}

/*
window.onload = function() {
	var vidstream = document.getElementById("stream");
	var config = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] };
	playStream(vidstream, null, null, null, config, function (errmsg) { console.error(errmsg); });
};*/