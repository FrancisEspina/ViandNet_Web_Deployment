$(document).ready(function () {
  // When a file is selected, show the preview
  $("#image-input").change(function () {
    var resultDiv = $("#result");
    // resultDiv.empty();
    readURL(this);
  });

  $("#upload-form").submit(function (event) {
    event.preventDefault();
    var formData = new FormData(this);
    $.ajax({
      type: "POST",
      url: "/classify",
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        displayPredictions(response);
      },
    });
  });
});

function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $("#image-container-preview").html(
        '<img class="mb-2 rounded-5"src="' +
          e.target.result +
          '" alt="Selected Image">'
      );
    };

    reader.readAsDataURL(input.files[0]); // convert to base64 string
  }
}

function displayPredictions(predictions) {
  var resultDiv = $("#result");
  var max_pred = predictions["max_prediction"];
  resultDiv.empty();

  var predictionDiv = $('<div class="prediction">');

  var predDetails = $("<p class='text-light'>");
  predDetails.html(
    "<b>Prediction:</b> " +
      "<h1>" +
      predictions["pred_class"] +
      "</h1>" +
      "<h4>Probability: <span class='badge bg-secondary'>" +
      max_pred.toFixed(2) +
      "</span></h4>"
  );
  predictionDiv.append(predDetails);

  var canvas = $("<canvas class='text-light'>");
  canvas.attr("id", "barChart");
  predictionDiv.append(canvas);
  resultDiv.append(predictionDiv);

  var labels = predictions["labels"].slice(0, 5);
  var values = predictions["values"].slice(0, 5);
  var data = {
    labels: labels,
    datasets: [
      {
        label: "Probability",
        data: values,
        backgroundColor: "white",
        borderColor: "white", // Set border color to white
        borderWidth: 1, // Set border width
      },
    ],
  };
  var ctx = document.getElementById("barChart").getContext("2d");
  var barChart = new Chart(ctx, {
    type: "bar",
    data: data,
    options: {
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            color: "white", // Set y-axis ticks color to white
          },
          grid: {
            color: "gray", // Set y-axis grid color to white
          },
        },
        x: {
          ticks: {
            color: "white", // Set x-axis ticks color to white
          },
          grid: {
            color: "gray", // Set x-axis grid color to white
          },
        },
      },
      plugins: {
        title: {
          display: true,
          text: "Top 5 Predictions",
          color: "white", // Set title text color to white
        },
        legend: {
          labels: {
            fontColor: "white", // Set legend labels color to white
          },
        },
      },
      elements: {
        line: {
          borderColor: "white", // Set line color to white
        },
      },
    },
  });
}
