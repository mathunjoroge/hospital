<?php
//determining the sex for purposes of test
$patient=$_GET['patient'];
$result = $db->prepare("SELECT * FROM patients WHERE opno=:patient");
$result->BindParam(':patient', $patient);
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$sex=$row['sex'];
$b=$row['age'];

}
$now = date('Y-m-d');
$dob = date("Y-m-d", strtotime($b));  
$date1=date_create($dob);
$date2=date_create($now);
$diff=date_diff($date1,$date2);
$days=(float)$diff->format("%R%a");
if (($days>4380)&&($sex="male")) {
$sex=1;
}
if ((180 <= $days) && ($days <=4380))  {
$sex=3;
}

if(($days<=180)&&($sex="male")) {
$sex=4;
}

?>