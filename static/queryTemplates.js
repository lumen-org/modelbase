	var queryUrl="http://127.0.0.1:5000/sample_query";
    var modelbaseUrl="http://127.0.0.1:5000/webservice";	
	var templates = {
		"showHeader": 
		{  "SHOW": "HEADER",
		   "FROM": "car_crashes" },
		"clone":
		{  "FROM": "car_crashes",
			"MODEL": "*",
			"AS": "car_crashes_cp" }, 
		"condition": 
		{ "FROM": "car_crashes_cp",
		  "MODEL": "*",
		  "AS": "car_crashes_cp_cond",
		  "WHERE": [
			{"name": "no_previous", "operator": "EQUALS", "value": 11 }
		  ]
		}, 
		"marginalize": 
		{	"FROM": "car_crashes_cp",
			"MODEL": [
				"speeding",
				"alcohol",
				"total"
			],
			"AS": "car_crashes_cp_marg"
		},
		"model": 
		{  "MODEL": [ 
				"speeding",
				"alcohol"
			],
			"AS:":"car_crashes_speedAlc",
			"FROM": "car_crashes",
			"WHERE": [
				{"name": "no_previous", "operator": "EQUALS", "value": 10 }
			]
		},
		"predict":  
		{	"PREDICT": [
				"speeding",
				{
				  "name": "alcohol",
				  "aggregation": "average"
				},
				"total",
				{
				  "name": "total",
				  "aggregation": "density"
				}

				],
				"FROM": "car_crashes",
				"WHERE": [
				{
				  "name": "alcohol",
				  "operator": "EQUALS",
				  "value": 2.3
				}
			],
			"SPLIT BY": [
				{
				  "name": "speeding",
				  "split": "equidist"
				},
				{
				  "name": "total",
				  "split": "equidist"
				}
			]
		},
		"showModels": {	"SHOW": "MODELS" }
	}

    function onError(error, msg) {
        console.error("ERROR " + msg + " : " + error)
    }

    function onSuccess(data, msg) {
        console.log("SUCCESS " + msg + " : " + data)
    }
    
    function executeQuery(json) {
     d3.json(modelbaseUrl)
            .header("Content-Type", "application/json")
            .post(JSON.stringify(json), onQueryExecuted)
    }

    function onQueryFetched (error, json) {
        if (error) onError(error, "onQueryFetched")
        onSuccess(JSON.stringify(json), "onQueryFetched")
        executeQuery(json)
    }

    function onQueryExecuted (error, json) {
        if (error) {
        	onError(error, "onQueryExecuted")
        	queryOutputField[0][0].value = error.response
        	return
        }
        var resultStr = JSON.stringify(json, null, 2)
        onSuccess(resultStr, "onQueryExecuted")
        queryOutputField[0][0].value = resultStr
    }

    // load a json query. When loaded send the query
    d3.json(queryUrl).post( onQueryFetched )
    
    var queryInputField = d3.select("#jsonQueryInput")
    queryInputField.on("keydown", function () {
      let ev = d3.event
      let id = ev.key
      let code = ev.keyCode
      if (id === "Enter" && (ev.shiftKey || ev.ctrlKey)) {
        let json = JSON.parse( queryInputField[0][0].value )
        executeQuery(json)
      }      
    })
	

	var qif = queryInputField[0][0];
	var callbackMaker = (key) => {return () => qif.value = JSON.stringify(templates[key],null,2)}
	for (let key in templates) {
		d3.select("#" + key + "Button").on("click", callbackMaker(key) );
	}

    var submitButton = d3.select("#submitButton")    
    var queryOutputField = d3.select("#queryResult")    
    submitButton.on("click", function () {      
      try{
        let json = JSON.parse( queryInputField[0][0].value )
        console.log(json)
        executeQuery(json)
      }
      catch (e) {
        if (e.name === 'SyntaxError') 
          queryOutputField[0][0].value = "invalid query"
        else 
          throw(e)
      }
    })