var aftercartimglink = null;
var beforecartimglink = null;

var imagehash = null;
var reader = new FileReader();
function readURL(input) {
    if (input.files && input.files[0]) {

        var loadingImage = loadImage(
            input.files[0],
            function (img,data) {
                //loadImage.parseMetaData(input.files[0], function (data) {
                console.log(data.exif.get('Orientation'));
                //});

                if(img.type === "error") {
                    console.log("Error loading image " + imageUrl);
                } else {
                    img.id = 'beforecartimage';
                    $('#beforecartimage').replaceWith(img);

                    //same as FileReader onload
                    reader.onload = function(event){
                        var data = event.target.result;
                        var imagehash = CryptoJS.SHA256( data );
                        console.log('image hash value: ' + imagehash);
                        document.getElementById('imagehash').value = imagehash;
                    };

                    reader.readAsDataURL(input.files[0]);
                }
            },
            {
                orientation: true
                //meta: true not needed when orientation = true
                //maxWidth: 600
            }
        );
    }
}

$( document ).ready(function() {
    console.log( "ready!" );

    var urlParams = new URLSearchParams(window.location.search);

    if(urlParams.has('imgdst')){
        aftercartimglink = urlParams.get('imgdst');
    }

    if(urlParams.has('imgsrc')){
        beforecartimglink = urlParams.get('imgsrc');
    }

    if(aftercartimglink){
        $('#aftercartimage')
            .attr('src', aftercartimglink);

        $('#aftercartimagelink')
            .attr('href', aftercartimglink);
    }
    if(beforecartimglink){
        $('#beforecartimage')
            .attr('src', beforecartimglink);

        $('#beforecartimagelink')
            .attr('href', beforecartimglink);
    }

    /*$('#imagesubmitform')
    .ajaxForm({
        url : '/',
        success : function (response) {
            //alert("The server says: " + response);
            var serverresponse = JSON.parse(response);
            var cartimglink = serverresponse.imagelink;

            $('#aftercartimage')
                .attr('src', cartimglink);

            $('#aftercartimagelink')
                .attr('href', cartimglink);

            aftercartimglink = cartimglink;
        }
    });

    $('#imagesubmitform').on('submit', function(e) {
        console.log( "submit pressed!" );
        e.preventDefault(); // prevent native submit
        $(this).ajaxSubmit({})
    });*/
});

