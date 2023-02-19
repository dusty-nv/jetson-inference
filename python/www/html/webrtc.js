
 var connections = {};

 function reportError(msg) {
	console.log(msg);
 }
 
 function getLocalStream() {
   var constraints = {'audio': false, 'video': true};
   if (navigator.mediaDevices.getUserMedia) {
	return navigator.mediaDevices.getUserMedia(constraints);
   }
 }

 function onIncomingSDP(url, sdp) {
   console.log('Incoming SDP: (%%s)' + JSON.stringify(sdp), url);

   function onLocalDescription(desc) {
	console.log('Local description (%%s)\\n' + JSON.stringify(desc), url);
	connections[url].webrtcPeer.setLocalDescription(desc).then(function() {
	  connections[url].websocket.send(JSON.stringify({ type: 'sdp', 'data': connections[url].webrtcPeer.localDescription }));
	}).catch(reportError);
   }

   connections[url].webrtcPeer.setRemoteDescription(sdp).catch(reportError);

   if( connections[url].type == 'inbound' ) {
	connections[url].webrtcPeer.createAnswer().then(onLocalDescription).catch(reportError);
   }
   else if( connections[url].type == 'outbound' ) {
	getLocalStream().then((stream) => {
	  console.log('Adding local stream');
	  connections[url].webrtcPeer.addStream(stream);
	  connections[url].webrtcPeer.createAnswer().then(onLocalDescription).catch(reportError);
	});
   }

 }


 function onIncomingICE(url, ice) {
   var candidate = new RTCIceCandidate(ice);
   console.log('Incoming ICE (%%s)\\n' + JSON.stringify(ice), url);
   connections[url].webrtcPeer.addIceCandidate(candidate).catch(reportError);
 }


 function onAddRemoteStream(event) {
   var url = event.srcElement.url;
   console.log('Adding remote stream to HTML video player (%%s)', url);
   connections[url].videoElement.srcObject = event.streams[0];
   connections[url].videoElement.play();
 }


 function onIceCandidate(event) {
   var url = event.srcElement.url;

   if (event.candidate == null)
	return;

   console.log('Sending ICE candidate out (%%s)\\n' + JSON.stringify(event.candidate), url);
   connections[url].websocket.send(JSON.stringify({'type': 'ice', 'data': event.candidate }));
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
	  console.log('WebRTC connection state (%%s) ' + connections[url].webrtcPeer.connectionState, url);
	}

	if( connections[url].type == 'inbound' )
	  connections[url].webrtcPeer.ontrack = onAddRemoteStream;
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

 function sendStream(hostname, port, path, configuration, reportErrorCB) {
   var l = window.location;
   if( path == 'null' )
	 return;
   if( l.protocol != 'https:' ) {
	 alert('Please use HTTPS to enable the use of your browser webcam');
	 return;
   }
   if( !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia ) {
	 alert('getUserMedia() not available (confirm HTTPS is being used)');
	 return;
   }
   var wsProt = (l.protocol == 'https:') ? 'wss://' : 'ws://';
   var wsHost = (hostname != undefined) ? hostname : l.hostname;
   var wsPort = (port != undefined) ? port : l.port;
   var wsPath = (path != undefined) ? path : '/ws';
   if (wsPort)
	wsPort = ':' + wsPort;
   var wsUrl = wsProt + wsHost + wsPort + wsPath;
   console.log('Video server URL: ' + wsUrl);
   var url = wsUrl;

   connections[url] = {};

   connections[url].type = 'outbound';
   connections[url].webrtcConfig = configuration;
   reportError = (reportErrorCB != undefined) ? reportErrorCB : function(text) {};

   connections[url].websocket = new WebSocket(wsUrl);
   connections[url].websocket.addEventListener('message', onServerMessage);
 }
