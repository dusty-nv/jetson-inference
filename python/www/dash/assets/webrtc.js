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


function playStream(videoElement, hostname, port, path, configuration, reportErrorCB) {
	var l = window.location;
	var wsHost = (hostname != undefined) ? hostname : l.hostname;
	var wsPort = (port != undefined) ? port : 49567; //l.port;
	var wsPath = (path != undefined) ? path : "my_stream";
	if (wsPort)
	wsPort = ":" + wsPort;
	var wsUrl = "wss://" + wsHost + wsPort + "/" + wsPath;
	console.log("video stream URL: " + wsUrl);

	html5VideoElement = videoElement;
	webrtcConfiguration = configuration;
	reportError = function (errmsg) { console.error(errmsg); };

	websocketConnection = new WebSocket(wsUrl);
	websocketConnection.addEventListener("message", onServerMessage);
}

if (!window.dash_clientside) {
    window.dash_clientside = {};
}

window.dash_clientside.webrtc = {
    // this function gets called from the dash app
    playStream: function(stream_config) {
		console.log('window.dash_clientside.webrtc.playStream() => ' + stream_config)
	    
		var video_player = document.getElementById("video_player");
		console.log('video player: ' + video_player)
		
		console.log('window location:  ' + window.location)
    }
}

/*
window.onload = function() {
	var vidstream = document.getElementById("stream");
	var config = { 'iceServers': [{ 'urls': 'stun:stun.l.google.com:19302' }] };
	playStream(vidstream, null, null, null, config, function (errmsg) { console.error(errmsg); });
};*/