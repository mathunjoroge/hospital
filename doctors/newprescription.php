<?php 
require_once('../main/auth.php');
include('../connect.php');
?>
<link rel="stylesheet" href="drugstyle.css">
<?php
if (!empty($_REQUEST["search"])) {
$search=$_REQUEST['search'];
$result = $db->prepare("SELECT * FROM patients WHERE opno=:o");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$a=$row['name'];
$b=$row['age'];
$c=$row['sex'];
$d=$row['opno'];

?>
<!DOCTYPE html>
<html>
<title>prescription for <?php echo $a; ?></title>
<?php
include "../header.php";
?>

</head>
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
<li class="breadcrumb-item active" aria-current="page">oncology patient</li>
<li class="breadcrumb-item active" aria-current="page">new prescription</li>


<li class="breadcrumb-item active" aria-current="page"><?php echo $a; ?>, &nbsp;<?php include "age.php"; ?>  &nbsp; sex: <?php echo $c; ?></li>
<?php }} ?>
</ol>
</nav>
<?php 
$result = $db->prepare("SELECT * FROM vitals JOIN patients ON vitals.pno=patients.opno WHERE pno=:o  ORDER BY vitals.id DESC LIMIT 1");
$result->BindParam(':o', $search);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$h=$row['height'];
$j=$row['weight'];
}
if (isset($j) && is_numeric($j) && is_numeric($h) ) {
$bsi=sqrt(($h*$j)/3600);
}
if (isset($j) && is_numeric($j) ) {
	
	$weight=$j;
	# code...
}
?>
<div class="container">

<span>
<!-- set this div to show when patient name is set -->
<?php
if (isset($_REQUEST["search"])) {
 	  ?>
<div class="container">
<form action="saveopdprescription.php" method="POST">
<span><select id='drug' style='width: 20%;' name="drug" data-live-search="true" required/>
<option value='0' ></option>
</select>&nbsp;
<input type="text" name="strength" placeholder="enter strength" type="number" style="width: 12%;"  required/>
<select name="units"><option value="" selected disabled>Please select units<optgroup label="general"><option>mg</option><option>units</option><option>mL</option><option>tablets</option><option>capsules</option></optgroup><optgroup label="kg or M&sup2;"><option>mg/kg</option><option>mg/M&sup2;</option></optgroup></select>
<select name="roa" placeholder="ROA"><option value="" selected disabled>route</option><option value="1">P.O</option><option value="2">IV</option><option value="3">IM</option><option value="4">SC</option><option value="5">topical</option><option value="6">sublingual</option><option value="7">per vaginal</option><option value="8">per rectal</option></select>
<input name="freq" type="text" placeholder="enter frequency" style="width: 10%;" required/>&nbsp;<input type="text" name="duration" placeholder="enter duration" required>
<input type="hidden" name="pn" value="<?php echo $search; ?>">
<input type="hidden" name="code" value="<?php echo $_REQUEST['code']; ?>">
<button class="btn btn-success">add</button>
<div class="suggestionsBox" id="suggestions" style="display: none;">
<div class="suggestionList" id="suggestionsList"> </form>
&nbsp; </div></div></div>
</hr>
<?php
$code=$_REQUEST['code'];
$result = $db->prepare("SELECT* FROM prescribed_meds WHERE code=:o");
$result->BindParam(':o', $code);
$result->execute(); 
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$trigger=$row['code'];
if (isset($trigger)) {    # code...


?>
<div class="container">
<table class="table-bordered" style="width: 100%;">
<tr>
<th>drug</th>
<th>route</th>
<th>strength</th>
<th>frequency</th>
<th>duration</th>
<th>delete</th>
</tr>
<?php 
     $code=$_GET['code'];
	   if ($useFdaDrugsList == 1) {
			$result = $db->prepare("SELECT ActiveIngredient, DrugName,duration,frequency,code,prescribed_meds.id AS id,prescribed_meds.strength AS strength,roa FROM prescribed_meds RIGHT OUTER JOIN meds ON prescribed_meds.drug=meds.id  WHERE code=:o");
	   }
		else {
			$result = $db->prepare("SELECT generic_name AS ActiveIngredient, brand_name AS DrugName,duration,frequency,code,prescribed_meds.id AS id,prescribed_meds.strength AS strength,roa FROM prescribed_meds RIGHT OUTER JOIN drugs as meds ON prescribed_meds.drug=meds.drug_id  WHERE code=:o");
		}
$result->BindParam(':o', $code);
        $result->execute();
        for($i=0; $row = $result->fetch(); $i++){
        $genericname=$row['ActiveIngredient'];
        $brandname=$row['DrugName'];
        $roa=$row['roa'];
        $strength=$row['strength'];
        $duration=$row['duration'];
        $frequency=$row['frequency'];
        $code=$row['code'];  
        $id=$row['id'];   
     ?>
     <tr>
       <td><?php echo $genericname; ?> (<?php echo $brandname; ?>)</td>
       <td><?php if ($roa==1) {echo "oral";
         # code...
       } 
       if ($roa==2) {echo "iv";
         # code...
       }
       if ($roa==3) {echo "IM";
         # code...
       }
       if ($roa==4) {echo "SC";
         # code...
       }
       if ($roa==5) {echo "topical";
         # code...
       }
        if ($roa==6) {echo "sublingual";
         # code...
       }
        if ($roa==7) {echo "per vaginal";
         # code...
       }
        if ($roa==8) {echo "per rectal";
         # code...
       }
       ?></td>
       <td><?php echo $strength; ?></td>
       <td><?php


if ($frequency==0) {
    echo "STAT";
} else {
    echo $frequency;
}
?></td>
       <td><?php echo $duration; ?> </td>
       <td><a  href="delete.php?id=<?php echo $id; ?>&code=<?php echo $code; ?>&search=<?php echo $search; ?>&dept=1"><button class="btn btn-danger">delete</button></a></td><?php } ?>
     </tr>
   </table>
   <p>&nbsp;</p>
   <a href="index.php?response=1&search=%20"><button class="btn btn-success" style="width: 70%;">save</button></a>
</div>
<?php }}} ?></div></div></div></div></div></div>
<script>

$(document).ready(function(){

$("#drug").select2({
placeholder:"find drug",
minimuminputLength:3,
theme: "classic",
ajax: {
url: "autosuggestdrug.php?q=term",
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