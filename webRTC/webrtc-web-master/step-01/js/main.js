'use strict';

// 미디어를 얻기위해 video, audio 결정
const mediaStreamConstraints = {
  video: true,
};

// Video element where stream will be placed.
const localVideo = document.querySelector('video');

// Local stream that will be reproduced on the video.
let localStream;

// 성공하면 srcObject속성을 통헤 MediaStream 반환
function gotLocalMediaStream(mediaStream) {
  localStream = mediaStream;
  localVideo.srcObject = mediaStream;
}

// Handles error by logging a message to the console with the error message.
function handleLocalMediaStreamError(error) {
  console.log('navigator.getUserMedia error: ', error);
}

// getUserMedia 호출 후 브라우저는 사용자에게 카메라 엑세스 권한 요청 
navigator.mediaDevices.getUserMedia(mediaStreamConstraints)
  .then(gotLocalMediaStream).catch(handleLocalMediaStreamError);
