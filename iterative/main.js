const fetchJson = (url, options = {}) => {
  return fetch(url, options).then((resp) => {
    if (!resp.ok) {
      throw new Error(`Response was not successful. ${resp.error}`);
      window;
    }

    return resp.json();
  });
};

const $ = document.querySelector;
const h = document.createElement;

// Create WebSocket connection.
const socket = new WebSocket("ws://localhost:8765");

// Connection opened
socket.addEventListener("open", (event) => {
  // socket.send(
  //   JSON.stringify({
  //     req_from: "client",
  //     type: "caption_new",
  //     payload: { new_caption: "hello" },
  //   }),
  // );
  socket.send(JSON.stringify({ hello: "world" }));
});

// Listen for messages
socket.addEventListener("message", (event) => {
  console.log("Message from server ", event.data);
});

document.querySelector("#caption_form").addEventListener("submit", (e) => {
  e.preventDefault();

  const caption = document.querySelector("#caption").value;
  const image_name = document.querySelector("#image_name").value;
  console.log(caption, image_name);

  console.log("submitted");
  socket.send(
    JSON.stringify({
      req_from: "client",
      type: "caption_new",
      payload: { new_caption: caption, image: image_name },
    }),
  );
});

// get the current caption image and text
// get the current loss
// get the statistics of the current run

// submit a new caption
// submit that the run should continue x number of steps
