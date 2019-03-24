function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#beforecartimage')
                .attr('src', e.target.result);

            $('#beforecartimagelink')
                .attr('href', e.target.result);
        };

        reader.readAsDataURL(input.files[0]);
    }
}

var aftercartimglink = null;
var beforecartimglink = null;

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

