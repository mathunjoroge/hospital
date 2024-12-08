<!DOCTYPE html>
<html>
<title>pharmacy</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
<script src="https://use.fontawesome.com/5c87075a3d.js"></script>
  <link href="../css/vertical-responsive-menu.min.css" rel="stylesheet">
  <link href="../css/demo.css" rel="stylesheet">
  <link rel="stylesheet" href="../css/bootstrap.min.css">
  <link rel="stylesheet" href="../css/bootstrap-select.css">
  <script src="../js/jquery.min.js"></script>
  <script src="../js/bootstrap.min.js"></script>
  <script src="../js/bootstrap-select.js"></script>
  <link href="../src/facebox.css" media="screen" rel="stylesheet" type="text/css" />
  <link rel="stylesheet" href="../css/jquery-ui.css">
  <script src="../js/jquery-1.12.4.js"></script>
  <script src="../js/jquery-ui.js"></script>
  <script src="../src/facebox.js" type="text/javascript"></script>
  <link href='../css/select2.min.css' rel='stylesheet' type='text/css'>
<!-- select2 script -->
<script src='../js/select2.min.js'></script>
</head>
<body>
<div class="container"> 
<div class="jumbotron" style="margin-top:6em;"> 
<div class="container">
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php">Home</a></li>
    <li class="breadcrumb-item active" aria-current="page">drugs reference</li>
    <li class="breadcrumb-item active" aria-current="page">details for <?php echo $_GET['search_q']; ?></li>
    <li class="breadcrumb-item" style="float: right;"><a href="interractions.php"> check drug interractions</a></li> 
    <li class="breadcrumb-item" style="float: right;"><a href="diseases.php">drugs and diseases</a></li>   
  </ol>
</nav>
<select id='patient' style='width: 40%;'  name="search" data-live-search="true"  required/>
<option value='0' ></option>
</select> 
<script>
$(document).ready(function(){
$("#patient").select2({
placeholder:"enter patient name or number",

minimuminputLength:3,
theme: "classic",
ajax: {
url: "../doctors/patient.php?q=term",
dataType: 'json',
type: "POST",
delay: 250,
data: function (params) {
return {
q: params.term, // search term
};
},
processResults: function (data) {
return {
results: data
};
},
cache: true
}
});
});
</script> 
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</form>
<script>
$(document).ready(function(){
$("#patient").select2({
placeholder:"enter patient name or number",
minimuminputLength:3,
theme: "classic",
ajax: {
url: "get_drug.php?q=term",
dataType: 'json',
type: "POST",
delay: 250,
data: function (params) {
return {
q: params.term, // search term
};
},
processResults: function (data) {
return {
results: data
};
},
cache: true
}
});
});
</script>

</body>
</html>