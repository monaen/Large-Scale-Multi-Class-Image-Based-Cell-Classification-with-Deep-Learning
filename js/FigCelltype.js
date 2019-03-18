// set the margin of the SVG
var margin = {top: 50, right: 50, bottom: 50, left: 100};

// set the width and height using the current width and height of the div
var width = 550 - margin.left - margin.right;
var height = 360 - margin.top - margin.bottom;

var figure = d3.select("#barplot_cell_samples").append("svg")
               .attr("width", width + margin.left + margin.right)
               .attr("height", height + margin.top + margin.bottom)
               .append("g")
               .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// define the x and y axes
var y = d3.scaleLinear()
          .range([height, 0])
          .domain([0,450000]);
var x = d3.scaleBand()
          .range([0, width])
          .domain(["THP1", "MCF7", "MB231", "PBMC"])
          .padding(0.36);

// plot the x and y coordinates
figure.append("g").call(d3.axisLeft(y));
figure.append("g").attr("transform", "translate(0," + height + ")").call(d3.axisBottom(x));


var data = [
  {preaugment: 200951, afteraugment: 256000, type: "THP1"},
  {preaugment: 374427, afteraugment: 378000, type: "MCF7"},
  {preaugment: 343394, afteraugment: 350000, type: "MB231"},
  {preaugment: 21223,  afteraugment: 321000, type: "PBMC"}
];

var colors = ["#ff9999", "#9999ff"];


// add dash line grid
figure.append("g")
      .attr("class", "grid")
      .style("stroke-dasharray", "3 3")
      .style("opacity", "0.36")
      // .attr("transform", "translate(0, ${height})")
      .call(d3.axisLeft()
              .scale(y)
              .tickSize(-width, 0, 0)
              .tickFormat("")
           )

// define the bar group
// const bargroup = figure.selectAll()
//                        .

// add preaugment bars
figure.selectAll()
      .data(data)      
      .enter()
      .append("rect")
      .attr("class", "bar")
      .style("fill", colors[0])
      .style("stroke", colors[0])
      .attr("x", function(d) { return x(d.type); })
      .attr("width", x.bandwidth() / 2.03)
      .attr("y", function(d) { return y(d.preaugment); })
      .attr("height", function(d) {return height - y(d.preaugment)})
      .on('mouseenter', function (s, i) {
        d3.select(this)
          .attr("opacity", 1)
          .transition()
          .duration(300)
          .attr("opacity", 0.6)
          .style("stroke", "red")

        const yvalue = y(s.preaugment)
        const xvalue = x(s.type)

        figure.append("line")
              // .transition()
              // .duration(300)
              .attr("id", "limit")
              .attr("x1", 0)
              .attr("y1", yvalue)
              .attr("x2", width)
              .attr("y2", yvalue)
              .attr("stroke", "red")
              .attr("stroke-dasharray", "5 5")

        figure.append("text")
              .attr("class", "labeltip")
              .attr("x", 0)
              .attr("y", yvalue - 5)
              .text(s.preaugment)
              .style("font-family", "Times")
              .style("font-weight", "bold")
              .style("font-style", "italic")
              .style("fill", "#ff7777")
      })
      .on('mouseleave', function (s, i) {
        d3.select(this)
          .attr("opacity", 0.6)
          .transition()
          .duration(300)
          .attr("opacity", 1)
          .style("stroke", colors[0])

        d3.selectAll("#limit").remove()
        d3.selectAll(".labeltip").remove()
      });


// add afteraugment bars
figure.selectAll()
      .data(data)      
      .enter()
      .append("rect")
      .style("fill", colors[1])
      .attr("x", function(d) { return x(d.type); })
      .attr("width", x.bandwidth() / 2)
      .attr("transform", "translate("+x.bandwidth() / 2+",0)")
      .attr("y", function(d) { return y(d.afteraugment); })
      .attr("height", function(d) {return height - y(d.afteraugment)})
      .on('mouseenter', function (s, i) {
        d3.select(this)
          .attr('opacity', 1)
          .transition()
          .duration(300)
          .attr('opacity', 0.6)
          .style("stroke", "blue")

        const yvalue = y(s.afteraugment)

        figure.append("line")
              .attr("id", "limit")
              .attr("x1", 0)
              .attr("y1", yvalue)
              .attr("x2", width)
              .attr("y2", yvalue)
              .attr("stroke", "blue")
              .attr("stroke-dasharray", "5 5")


        figure.append("text")
              .attr("class", "labeltip")
              .attr("x", 0)
              .attr("y", yvalue - 5)
              .text(s.afteraugment)
              .style("font-family", "Times")
              .style("font-weight", "bold")
              .style("font-style", "italic")
              .style("fill", "#7777ff")
      })
      .on('mouseleave', function (s, i) {
        d3.select(this)
          .attr('opacity', 0.6)
          .transition()
          .duration(300)
          .attr('opacity', 1)
          .style("stroke", colors[1])


        d3.selectAll("#limit").remove()
        d3.selectAll(".labeltip").remove()
      });


// add text label to y coordinates
figure.append("text")
      .attr("x", -(height / 2) - margin.top)
      .attr("y", -margin.left / 1.6)
      .attr("transform", "rotate(-90)")
      .text("Number of cells")
      .style("font-family", "Times")
      .style("font-weight", "bold")
      .style("font-size", "12pt");

// add text label to x coordinates
figure.append("text")
      .attr("x", (width - margin.left) / 2)
      .attr("y", height + margin.top / 1.2)
      .text("Cell types")
      .style("font-family", "Times")
      .style("font-weight", "bold")
      .style("font-size", "12pt");

// draw legend
figure.append("rect")
      .attr("class", "legend_background")
      .attr("x", 0)
      .attr("y", 10)
      .attr("width", 110)
      .attr("height", 50)
      .attr("fill", "#ffffff")
      .attr("transform", "translate("+ (width-margin.right-60) +", 0)")
      .attr("stroke", "#666666")
      .attr("stroke-width", 1)
      .attr('opacity', 0.6)

figure.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 10)
      .attr("height", 10)
      .attr("transform", "translate("+ (width-margin.right-52) +", 18)")
      .attr("fill", colors[0])
      .style("border-radius", 6)

figure.append("text")
      .attr("x", (width-margin.right-35))
      .attr("y", 28)
      .text("Pre-augment")
      .attr("fill", "#666666")
      .style("font-family", "Times")
      .style("font-weight", "bold")
      .style("font-size", "9pt");

figure.append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", 10)
      .attr("height", 10)
      .attr("transform", "translate("+ (width-margin.right-52) +", 42)")
      .attr("fill", colors[1])
      .style("border-radius", 6)

figure.append("text")
      .attr("x", (width-margin.right-35))
      .attr("y", 52)
      .text("After-augment")
      .attr("fill", "#666666")
      .style("font-family", "Times")
      .style("font-weight", "bold")
      .style("font-size", "9pt");



//  ====================================== visualization of cells ====================================== //

