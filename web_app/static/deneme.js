var tableIndex = 1;

function addValuesToTable(gen, data) {
  jQuery.each(data.data, function (index, value) {
    $("#results").append(
      "<tr>" +
        "<th scope='row'>" +
        tableIndex +
        "</th>" +
        "<td>" +
        gen +
        "</td>" +
        "<td>" +
        value.fitnessScore +
        "</td>" +
        "<td>" +
        value.loss +
        "</td>" +
        "<td>" +
        value.val_loss +
        "</td>" +
        "<td>" +
        value.mape +
        "</td>" +
        "<td>" +
        value.val_mape +
        "</td>" +
        "<td>" +
        value.mse +
        "</td>" +
        "<td>" +
        value.val_mse +
        "</td>" +
        "<td>" +
        value.optimizer +
        "</td>" +
        "<td>" +
        value.architecture.split("/").join("<br>") +
        "</td>" +
        "<td><button type='button' class='btn btn-outline-dark'><i class='bi-download'></i></button></td>" +
        "</tr>"
    );
    tableIndex++;
  });
  $("#total").text(data.data.length);
}

$("#search").on("click", function () {
  $("#search").attr("disabled", true);
  var generationCount = 0;
  $("#results").empty();

  var form_data = new FormData($("#upload-file")[0]);

  $.ajax({
    url: "/ga-results/" + true,
    type: "get",
    data: form_data,
    contentType: false,
    processData: false,
    success: function (response) {
      if (response.status == 1) {
        addValuesToTable(generationCount, data);

        while (generationCount < 1) {
          generationCount += 1;
          $.get("/ga-calback", function (data) {
            addValuesToTable(generationCount, data);

            $("#search").attr("disabled", false);
          });
        }
      } else {
        alert("File not uploaded");
      }
    },
  });

  /*$.get("/ga-results/" + true, function (data) {
    addValuesToTable(generationCount, data);

    while (generationCount < 1) {
      generationCount += 1;
      $.get("/ga-calback", function (data) {
        addValuesToTable(generationCount, data);

        $("#search").attr("disabled", false);
      });
    }
  });*/
});

var tableIndex2 = 1;

function addValuesToTable2(gen, data) {
  jQuery.each(data.data, function (index, value) {
    $("#results2").append(
      "<tr>" +
        "<th scope='row'>" +
        tableIndex2 +
        "</th>" +
        "<td>" +
        gen +
        "</td>" +
        "<td>" +
        value.average_accuracy +
        "</td>" +
        "<td>" +
        value.accuracy +
        "</td>" +
        "<td>" +
        value.val_accuracy +
        "</td>" +
        "<td>" +
        value.loss +
        "</td>" +
        "<td>" +
        value.val_loss +
        "</td>" +
        "<td>" +
        value.optimizer +
        "</td>" +
        "<td>" +
        value.architecture +
        "</td>" +
        "<td><button type='button' class='btn btn-outline-dark'><i class='bi-download'></i></button></td>" +
        "</tr>"
    );
    tableIndex2++;
  });
  $("#total2").text(data.data.length);
}

$("#search2").on("click", function () {
  $("#search2").attr("disabled", true);

  var generationCount = 0;
  $("#results2").empty();
  $.get("/ga-results/" + false, function (data) {
    addValuesToTable2(generationCount, data);

    while (generationCount < 1) {
      generationCount += 1;
      $.get("/ga-calback", function (data) {
        addValuesToTable2(generationCount, data);

        $("#search2").attr("disabled", false);
      });
    }
  });
});
