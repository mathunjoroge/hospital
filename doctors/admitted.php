<?php 
require_once('../main/auth.php');
include ('../connect.php');
?>
<!DOCTYPE html>
<html>
<title>doctors</title>
<?php
include "../header.php";
?>
</head>
<script>
    $('select').select2();

// Changing specific search field
$('.select2').click(function(){
  $('#select2-bap-results')
    .parent()
    .siblings('.select2-search')
    .css('background-color', 'black');
});
</script>
<style>
#select2-disease-results { background-color: teal; }

/* Higlighted (hover) result */
#select2-disease-results .select2-results__option--highlighted { background-color: blue! important; }

/* Selected option */
#select2-disease-results .select2-results__option[aria-selected=true] { background-color: black !important; }
/* Each Result */
.select2-container--default .select2-selection--multiple .select2-selection__choice {
    background-color: teal;
}
/* style for ddx */
#select2-ddx-results { background-color: teal; }

/* Higlighted (hover) result */
#select2-ddx-results .select2-results__option--highlighted { background-color: blue! important; }

/* Selected option */
#select2-ddx-results .select2-results__option[aria-selected=true] { background-color: black! important; }
/* Each Result */
.select2-container--default .select2-selection--multiple .select2-selection__choice {
    background-color: blue;
}


</style>

<body>
<header class="header clearfix" style="background-color: #3786d6;">
</button>
<?php include('../main/nav.php'); ?>   
</header><?php include('side.php'); ?>

<div class="content-wrapper" style=" background-image: url('../images/doctor.jpg');">
<div class="jumbotron" style="background: #95CAFC;">
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">out patient</li>

<?php
if (isset($_GET['search'])) {
$search=$_GET['search'];
$pn=$_GET['search'];
$response=0;
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
$b=$row['age'];
$c=$row['sex'];
$d=$row['opno'];
?>
<li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?> 
<?php
include 'age.php';
?>
<?php } ?>
</ol>
</nav>
<form action="savenotes.php" method="POST">
<input type="hidden" name="pn" value="<?php echo $pn; ?> ">      
<label>add clinical notes</label></br> 
<textarea name="notes" style="width: 70%;height:15em;"></textarea></br>
<div class="container" style="width: 70%;margin-left:-10px;"> 
<div class="row"> 
<div class="col-sm-6">   
<label>request lab tests</label></br>
<select id="maxOption2" class="selectpicker form-control" name="lab[]" multiple/>
<option value="" disabled="">-- Select test--</option>
<?php 
$result = $db->prepare("SELECT * FROM lab_tests");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
echo "<option value=".$row['id'].">".$row['name']."</option>";
}

?>      
</select></br>
</div> 
<div class="col-sm-6"> 
<label>request imagings</label></br>
<select id="maxOption3" style="width: 50%;" class="selectpicker form-control"  name="image[]" multiple/>
<option value="" disabled="">-- Select imaging--</option>
<?php 
$result = $db->prepare("SELECT * FROM imaging");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
echo "<option value=".$row['imaging_id'].">".$row['imaging_name']."</option>";
}

?> 

</select>
</div>
</div>
</div>
<div class="container" style="width: 70%;margin-left:-10px;">
<label>diagnosis</label></br> 
<select id='disease' style='width: 90%;' name="dx[]" data-live-search="true" class="form-interests-input-field"   multiple>
</select>
</div>
<div class="container">
<p>&nbsp;</p>
<button class="btn btn-success btn-large" style="width: 65%;">save</button></br></form> </div>

<p>&nbsp;</p>
<a href="prescribe_inp.php?search=<?php echo $search; ?>&code=<?php echo rand(); ?>&response=0"><button class="btn btn-success"> prescribe drugs</button></a><?php } ?>
</div> 
<div>
    
</div> 
</div>  


<?php
$respose=$_GET['response'];

if ($respose==1) {

?>
<script> 
  setTimeout(function(){
    $("#message").hide();
  }, 10000);
</script>
<?php if($_GET["response"]==1){ ?>
<div id="message">
<p class="text-left text-success">prescription has been saved</p>
</div>
<?php } }?>
<script>

$(document).ready(function(){

$("#disease").select2({
placeholder:"select diagnosis, multiple allowed",
minimuminputLength:3,

ajax: {
url: "diseases.php?q=term",
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