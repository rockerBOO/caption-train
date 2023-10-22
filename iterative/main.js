const fetchJson = (url, options={}) => {
	return fetch(url, options).then(resp => {
		if (!resp.ok) {
			throw new Error(`Response was not successful. ${resp.error}`);
		}

		return resp.json()
	})
}

const $ = document.querySelector;
const h = document.createElement;



// get the current caption image and text
// get the current loss
// get the statistics of the current run

// submit a new caption
// submit that the run should continue x number of steps

