<?php
require_once "../main/auth.php"; ?>
<!DOCTYPE html>
<html>
<title>doctors</title>
<?php include "../header.php"; ?>
</head>

<body>
    <style>
    /* style for background*/
   #disease + .select2-selection__choice
{
	background-color: blue;

}
    /* Input field */
#disease + .select2 .select2-selection__rendered {  background-color:black; }

/* Each result */
#select2-disease-results { background-color: teal; }

/* Higlighted (hover) result */
#select2-disease-results .select2-results__option--highlighted { background-color: #0033cc! important; }

/* Selected option */
#select2-disease-results .select2-results__option[aria-selected=true] { background-color: blue !important; }


// These 2 are special they would require js if you dont want to change the style for each select 2
/* Around the search field */
.select2-search { background-color: orange; }

/* Search field */
.select2-search input { background-color: pink; }
.select2-container--default .select2-selection--multiple .select2-selection__choice {
background-color: blue;
}
</style>
<style>
    /* Input field */
#ddx + .select2 .select2-selection__rendered {  background-color:black; }

/* Each result */
#select2-ddx-results { background-color: black; }

/* Higlighted (hover) result */
#select2-ddx-results .select2-results__option--highlighted { background-color: #0033cc! important; }

/* Selected option */
#select2-ddx-results .select2-results__option[aria-selected=true] { background-color: blue !important; }


// These 2 are special they would require js if you dont want to change the style for each select 2
/* Around the search field */
.select2-search { background-color: orange; }

/* Search field */
.select2-search input { background-color: pink; }
</style>

<header class="header clearfix" style="background-color: #3786d6;">
</button>
<?php include "../main/nav.php"; ?>   
</header><?php include "side.php"; ?>

<div class="content-wrapper" style=" background-image: url('../images/doctor.jpg');">

<div class="jumbotron" style="background: #95CAFC;">
<nav aria-label="breadcrumb" style="width: 90%;">
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">out patient</li>
<li><a rel="facebox" href='disease_add.php'>add disease</a></li>
<?php if (!empty($_GET["search"])) {

    $search = $_GET["search"];
    include "../connect.php";
    $result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
    $result->BindParam(":o", $search);
    $result->execute();
    for ($i = 0; ($row = $result->fetch()); $i++) {

        $a = $row["name"];
        $b = $row["age"];
        $c = $row["sex"];
        $d = $row["opno"];
        ?>
<li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?> 
<?php include "age.php"; ?>
<?php
    }
    ?>

</ol>
</nav>
<form action="index.php?" method="GET">
<span><?php include "../pharmacy/patient_search.php"; ?> 
<input type="hidden" name="response" value="0"> <button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</form>

<?php
$search = $_GET["search"];
$nothing = "";
if ($search != $nothing) {
    # code...
}
?>
<?php
$search = $_GET["search"];
$response = 0;
include "../connect.php";
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {

    $a = $row["name"];
    $b = $row["age"];
    $c = $row["sex"];
    $d = $row["opno"];
    ?>


<h3 align="center">patient medical information</h3>
<script>
$(document).ready(function(){
$("#hide").click(function(){
$("#vitals").fadeToggle('swing');    
});
});
</script>
<div class="container">
<button id="hide">Click to see/hide vitals</button><br>
</div>
<div class="container" id="vitals">
<?php
$result = $db->prepare(
    "SELECT * FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o"
);
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {

    $e = $row["systolic"];
    $f = $row["diastolic"];
    $g = $row["rate"];
    $h = $row["height"];
    $j = $row["weight"];
    $k = $row["temperature"];
    $l = $row["breat_rate"];
    $search = $search;
    $rbs = $row["rbs"];
    $date = $row["date"];
    $spo = $row["spo"];
    ?>
<div class="container-fluid">
<?php if (isset($e)) { ?>
<table class="table table-bordered">
<tr>
<th>date</th>
<th>systolic</th>
<th>diastolic</th>
<th>rate</th>
<th>SPO<sub>2</sub></th>
<th>comments</th>
</tr>
<?php
$result = $db->prepare(
    "SELECT systolic, diastolic,rate FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o AND  systolic > ''"
);
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {

    $e = $row["systolic"];
    $f = $row["diastolic"];
    $g = $row["rate"];
    ?>
<tr>
<td><?php echo $date; ?></td>
<td><?php echo $e; ?></td>
<td><?php echo $f; ?></td>
<td><?php echo $g; ?></td>
<td><?php echo $spo; ?></td>
<?php
if ($e < 90 || $f < 60) {
    $alert =
        "the patient is hypotensive, rapid action is needed," .
        "</br>" .
        " as this my lead to renal failure or even death!";
}
if ((90 <= $e && $e <= 119) || (60 <= $f && $f <= 80)) {
    $alert = "blood pressure is normal";
}
if ((121 <= $e && $e <= 139) || (81 <= $f && $f <= 89)) {
    $alert = "the patient is prehypertensive";
}
if ((140 <= $e && $e <= 159) || (90 <= $f && $f <= 99)) {
    $alert = "patient in stage 1 hypertension, action needed";
}
if ($e >= 160 || $f >= 100) {
    $alert = "patient in stage 2 hypertension,action needed";
}
$haystack = $alert;
$needle = "needed";
if (strpos($haystack, $needle) !== false) {
    $myclass = "alert alert-danger";
}
if (strpos($haystack, $needle) == false) {
    $myclass = "alert alert-success";
}
?>
<td class="<?php echo $myclass; ?>"> <?php echo $alert; ?> </td>
<?php
}
?>
</tr>
</table>
<?php } ?>
<?php if (isset($rbs)) { ?>
<table class="table table-bordered">
<tr>
<th>date</th>
<th>rbs</th>
<th>comments</th>
</tr>
<?php
$result = $db->prepare(
    "SELECT * FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o AND  rbs > ''"
);
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {
    $rbs = $row["rbs"]; ?>
<tr>
<td><?php echo $date; ?></td>
<td><?php echo $rbs; ?></td>
<?php
if ($rbs < 4.4) {
    $message = "patient is hypoglycemic";
    $class = "alert alert-danger";
}
if ($rbs >= 4.4 && $rbs <= 7.8) {
    $message = "rbs is normal";
    $class = "alert alert-success";
}
if ($rbs >= 7.8 && $rbs <= 11.1) {
    $message = "patient is prediabetic";
    $class = "alert alert-warning";
}
if ($rbs > 11.1) {
    $message = "patient is diabetic";
    $class = "alert alert-danger";
}
?>
<td class="<?php echo $class; ?>"> <?php echo $message; ?></td>
</tr><?php
}
?>
</table>
<?php } ?>
<?php if (isset($k)) { ?>
<table class="table table-bordered">
<tr>
<th>date</th>
<th>temperature</th>
<th>comments</th>
</tr>
<?php
$result = $db->prepare(
    "SELECT * FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o AND  temperature > ''"
);
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {
    $k = $row["temperature"]; ?>
<tr>
<td><?php echo $date; ?></td>
<td><?php echo $k; ?>&#x2103;</td>
<?php
if ($k < 36.1) {
    $tmessage = "patient is hypothermic";
    $tclass = "alert alert-danger";
}
if ($k >= 36.1 && $k <= 38) {
    $tmessage = "temperature is normal";
    $tclass = "alert alert-success";
}
if ($k > 38) {
    $tmessage = "patient has fever";
    $tclass = "alert alert-danger";
}
?>
<td class="<?php echo $tclass; ?>"> <?php echo $tmessage; ?></td>
</tr><?php
}
?>
</table>
<?php } ?>  
<?php if (isset($l)) { ?>
<table class="table table-bordered">
<tr>
<th>date</th>
<th>breath rate</th>
<th>comments</th>
</tr>
<?php
$result = $db->prepare(
    "SELECT * FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o AND  breat_rate > ''"
);
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {
    $l = $row["breat_rate"]; ?>
<tr>
<td><?php echo $date; ?></td>
<td><?php echo $l; ?> bpm</td>
<?php
if ($l < 12) {
    $bmessage = "patient is hypopnic";
    $bclass = "alert alert-danger";
}
if ($l >= 12 && $l <= 25) {
    $bmessage = "normal breathing rate";
    $bclass = "alert alert-success";
}
if ($l > 25) {
    $bmessage = "patient has tarchypnic";
    $bclass = "alert alert-danger";
}
?>
<td class="<?php echo $bclass; ?>"> <?php echo $bmessage; ?></td>
</tr><?php
}
?>
</table>
<?php } ?> <?php if (isset($h)) { ?>
<table class="table table-bordered">
<tr>
<th>date</th>
<th>weight</th>
<th>height</th>
<th>BSA</th>
<th>BMI</th>
</tr>
<?php
$result = $db->prepare(
    "SELECT * FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o"
);
$result->BindParam(":o", $search);
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {

    $j = $row["weight"];
    $h = $row["height"];
    if (is_numeric($h)) {
        $hm = $h / 100;
        $hms = pow(2, $hm);
    }
    if (is_numeric($j) && is_numeric($h)) {
        $bsi = sqrt(($h * $j) / 3600);
    }
    ?>
<tr>
<td><?php echo $date; ?></td>
<td><?php if (isset($j)) {
    // code...
 echo $j.' kg';}
  else{
    echo 'weight not taken';
 } ?></td>
<td><?php if (is_numeric($h)) {
    # code...
    echo $h;
} ?> cm</td> 
<td><?php
 if (is_numeric($h)) { 
if ((isset($h))&& isset($j) ){
    $bsa= (round(sqrt((($h*$j)/3600)),3));
echo $bsa;
} }?> M&sup2;</td>
<td><?php  if (is_numeric($h)) { 
if ((isset($h))&& isset($j) ){
    $hsq=($h/100)*($h/100);
       $bmi=($j/$hsq);
echo round(($bmi),3);
} } ?></td>
</tr><?php
}
?>
</table>
<?php } ?>
</div>
</div>
<?php
}
?>
<form action="savepatient.php" method="POST">
<input type="hidden" name="pn" value="<?php echo $search; ?> ">
<div class="container" style="width: 80%;">
<div class="container">
<label>chief complaint</label></br>
<input type="text" name="cc" style="width:40%;" class="form">
</div>
<div class="row">
<div class="col-sm-6">
<label>history of presenting illness</label></br> 
<textarea name="hpi" style="width: 90%;height:15em;" placeholder="description of the illness....." ></textarea></br>
   </div>
<div class="col-sm-6">
<label>on examination.</label></br> 
<textarea name="physical_examination" style="width: 90%;height:15em;" placeholder="on examination ........" required/></textarea></br>
</div>
</div>
<div class="row">
<div class="col-sm-6">
<label>request lab tests</label></br>
<select id="maxOption2" class="selectpicker show-menu-arrow form-control" data-live-search="true" title="Please select lab tests" name="lab[]" multiple >
<option value="" disabled="">-- Select test--</option><?php
$result = $db->prepare("SELECT * FROM lab_tests");
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {
    echo "<option value=" . $row["id"] . ">" . $row["name"] . "</option>";
}
?>      
</select>
</div>
<div class="col-sm-6">
<label>request imagings</label></br>
<select  class="selectpicker show-menu-arrow form-control" data-live-search="true" title="Please select imaging" name="image[]" multiple>
<option value="" disabled="">-- Select imaging--</option><?php
$result = $db->prepare("SELECT * FROM imaging");
$result->execute();
for ($i = 0; ($row = $result->fetch()); $i++) {
    echo "<option value=" .
        $row["imaging_id"] .
        ">" .
        $row["imaging_name"] .
        "</option>";
}
?>      
</select> 
</div>
</div>
<div class="container">
<label>diagnosis</label></br>             
<select id='disease' style='width: 105%;' name="dx[]" data-live-search="true"  multiple>

</select>
<script>

$(document).ready(function(){

$("#disease").select2({
placeholder:"find disease",
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
<script>

$(document).ready(function(){

$("#ddx").select2({
placeholder:"find disease",
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
</div>    
</div>
<div class="container">
<p>&nbsp;</p>
<button class="btn btn-success btn-large" style="width: 80%;margin-left: 10%;" >save</button></div></form><?php
}
?></div>
</div>
<?php
$respose = $_GET["response"];
if ($respose == 1) { ?>
<div class="alert alert-success" style="width: 20%;margin-left: 20%;"><p> patient data saved successifully</p></div>

</div><?php }
?>
</div>
</div>

</div></div></div>                
</div></div></div>


<?php
} ?>

</div></div>
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

</body>
</html>