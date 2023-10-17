
fetch("list.json").then(resp => {
    console.log(resp)
    if (resp.ok == false) { 
        throw new Error("Invaild response from the server", resp.statusText); 
    } 
    return resp.json();
}).then(json => {
    const main = document.querySelector("main")
    json.sort((a, b) => {
        const aImage = a['image'].toUpperCase();
        const bImage = b['image'].toUpperCase();

        if (aImage < bImage) {
            return -1;
        }

        if (aImage > bImage) {
            return 1;
        }

        return 0;
    });

    const results = json.map(item => {
        const figure = document.createElement('figure')
        figure.id = item.image
        const img = document.createElement('img');
        img.src = '/img/' + item.image
        img.title = item.image

        const caption = document.createElement('figcaption')
        caption.innerText = item.caption

        figure.appendChild(img)
        figure.appendChild(caption)

        return figure
    })



    results.forEach(result => {
        main.appendChild(result)
    })
}).catch(e => {
    console.error("Could not get the caption results", e)
});
