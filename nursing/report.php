<?php 
require_once('../main/auth.php');
include('../connect.php');
?> 
<!DOCTYPE html>
<html>
<title>MOH 204B</title>

<?php
include "../header.php";
?>
</head>
<header class="header clearfix" style="background-color: #3786d6;;">
<?php include('../main/nav.php'); ?>   
</header>
<?php include('side.php'); ?>

<script>
$( function() {
$( "#mydate" ).datepicker({
changeMonth: true,
changeYear: true
});
} );

</script>
<script>
$( function() {
$( "#mydat" ).datepicker({
changeMonth: true,
changeYear: true
});
} );
</script>  
<link rel="stylesheet" href="../main/jquery-ui.css">
<script src="../main/jquery-1.12.4.js"></script>
<script src="../main/jquery-ui.js"></script>
<div class="jumbotron" style="background: #95CAFC;">

<nav>
<ol class="breadcrumb">
<li class="breadcrumb-item"><a href="index.php?search= &response=0">Home</a></li>
<li class="breadcrumb-item active" aria-current="page">MOH 204B</li>
</li>

</ol>
</nav>
<div class="container" align="center">         
<form action="report.php" method="GET">
from: <input type="text" id="mydate"  name="d1" autocomplete="off" placeholder="pick start date" required/> to: <input type="text" id="mydat"  name="d2" autocomplete="off" placeholder="pick end date" required/>
<button class="btn btn-success"><i class="icon icon-save icon-large"></i>submit</button></span>     
</div>
</form>
<?php 
if (isset($_GET["d1"])) {
$d1=$_GET['d1']." 00:00:00"; 
$d2=$_GET['d2']." 23:59:59";
$date1=date("Y-m-d H:i:s", strtotime($d1));
$date2=date("Y-m-d H:i:s", strtotime($d2));
$d=0;
$e=3;

//end of pharmacy table


?>
<center><b>patients visits  from: <?php echo date("d-m-Y", strtotime($d1)); ?>  to: <?php echo date("d-m-Y", strtotime($d2)); ?> </b></center>


<?php 
$result = $db->prepare("SELECT p.name, p.age, p.sex AS sex, p.date AS date, p.opno, v.systolic, v.diastolic, v.rate, v.temperature, v.height, v.weight, GROUP_CONCAT(DISTINCT icdc.title) AS disease , GROUP_CONCAT(DISTINCT pm.drug) AS drug, GROUP_CONCAT(DISTINCT m.ActiveIngredient) AS active_ingredient 
FROM patients p
JOIN vitals v ON p.opno=v.pno
JOIN dx d ON p.opno=d.patient
JOIN icd_second_level_codes icdc ON d.disease=icdc.code
JOIN prescribed_meds pm ON p.opno=pm.patient
JOIN meds m ON pm.drug=m.id WHERE date(p.date)>=:a AND date(p.date)<=:b GROUP BY p.opno");
$result->bindParam(':a',$date1);
$result->bindParam(':b',$date2);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$b=$row['age'];
?>

<table class="table-bordered table-responsive" style = "word-wrap: break-word;">
<tr>
<th>Name</th>
<th>Age</th>
<th>Sex</th>
<th>Date</th>
<th>blood pressure mm/Hg</th>
<th>heart Rate</th>
<th>Temperature</th>
<th>Height</th>
<th>Weight</th>
<th>Disease</th>
<th>drug prescribed</th>
</tr>
<tr>
<?php
for($i=0; $row = $result->fetch(); $i++){
$b=$row['age'];
$c=$row['sex'];
echo "<tr>";
echo "<td>".$row['name']."</td>";
?>
<td><?php include '../doctors/age.php'; ?></td>
<td><?php echo $c; ?></td>
<?php
echo "<td>".$row['date']."</td>";
echo "<td>".$row['systolic']."/".$row['diastolic']."</td>";
echo "<td>".$row['rate']." BPM"."</td>";
echo "<td>".$row['temperature']."&#8451;"."</td>";
if (empty($row['height'])) {
echo "<td>height not available</td>";

}
else{
echo "<td>".$row['height']." cM"."</td>";
}
echo "<td>".$row['weight']." kg"."</td>";
echo "<td>".$row['disease']."</td>";
echo "<td>".$row['active_ingredient']."</td>";
echo "</tr>";
}
?>
</tr>
</table>
</div>
<?php }
}
?>
</div></div>
</br>


</body>
</html>