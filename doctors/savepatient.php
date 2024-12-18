<?php
include('../connect.php');
session_start();
$a=date('Y-m-d');
$doctor = $_SESSION['SESS_FIRST_NAME'];
$j = $_POST['pn'];
if(isset($_POST['physical_examination'])){
$physical = $_POST['physical_examination'];
//inserting data into physical examinations table
$sql = "INSERT INTO physicals (patient, description) VALUES (:a, :b)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $j,
':b' => $physical
));
}
if (isset($_POST['emergency'])) {
$emergency = $_POST['emergency'];
$a = date('Y-m-d H:i:s');
$sql = "INSERT INTO patient_notes (created_at,patient,notes,posted_by) VALUES (:a,:b,:c,:d)";
$q = $db->prepare($sql);
$q->execute(array(
':a' => $a,
':b' => $j,
':c' => $emergency,
':d' => $doctor
));
}
if (isset($_POST['lab'])) {
$labs = $_POST['lab'];
foreach ($labs as $lab) {
$sql = "INSERT INTO lab (test,opn,reqby) VALUES (:a,:b,:c)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $lab,
':b' => $j,
':c' => $doctor
));
}
}
if (isset($_POST['ddx'])) {
$ddxs = $_POST['ddx'];
foreach ($ddxs as $ddx) {
$sql = "INSERT INTO ddx (patient, disease) VALUES (:a, :b)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $j,
':b' => $ddx
));
}
}
if (isset($_POST['dx'])) {
$dxs = $_POST['dx'];
foreach ($dxs as $dx) {
$sql = "INSERT INTO dx (patient, disease) VALUES(:a,:b)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $j,
':b' => $dx
));
}
}
if (isset($_POST['image'])) {
$images = $_POST['image'];
foreach ($images as $image) {
$sql = "INSERT INTO req_images (test,opn,reqby) VALUES (:a,:b,:c)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $image,
':b' => $j,
':c' => $doctor
));
}
}
$p_exam= $_POST['physical_examination'];
$b   = $_POST['hpi'];
$d   = $_POST['cc'];
$sql = "INSERT INTO prescriptions (date,hpi,cc,opno,pexam) VALUES (:a,:b,:d,:j,:pexam)";
$q   = $db->prepare($sql);
$q->execute(array(
':a' => $a,
':b' => $b,
':d' => $d,
':j' => $j,
':pexam' => $p_exam
));

$reset = 0;
$sql   = "UPDATE patients
SET  served=?
WHERE opno=?";
$q     = $db->prepare($sql);
$q->execute(array(
$reset,
$j
));
 $has_bill =1;
$sql = "UPDATE patients
        SET  has_bill=?
		WHERE opno=?";
$q = $db->prepare($sql);
$q->execute(array($has_bill,$j)); 
$reset =3;
$sql   = "UPDATE patients
SET  served=?
WHERE opno=?";
$q     = $db->prepare($sql);
$q->execute(array(
$reset,
$j
));
?>
<?php
if (isset($_GET['resp'])) {
header("location: inpatient.php?search= &response=1");
# code...
}
if (!isset($_GET['resp'])) {
# code...
$code = rand();
header("location: newprescription.php?search=$j&response=0&code=$code");
}
?>