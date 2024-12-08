<?php
session_start(); 
include('../connect.php');
$id=$_POST['patient'];
$a=$_POST['clinic'];
$c=$_POST['total'];

$clinics=explode(",",$a);
foreach ($clinics as $clinic) {

?>
<?php 
$sql = "INSERT INTO clinic_fees (clinic_id, patient) VALUES ( ':a', ':b')";
$q = $db->prepare($sql);
$q->execute(array(':a'=>$clinic,':b'=>$id));
//save data into visits table
$sql = "INSERT INTO visits (patient) VALUES (:h)";
$q = $db->prepare($sql);
$q->execute(array(':h'=>$id));

$served=1;
$sql = "UPDATE patients
SET  served=?,
has_bill=?
WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($served,$served,$id));
header("location: pclinics.php?search= &response=4");
}
?>