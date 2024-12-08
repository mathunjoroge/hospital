<?php
include '../connect.php';

// Number of records fetch
$numberofrecords = 20;

if(isset($_POST['q'])){

// Fetch records
$search = $_POST['q'];
// check the drugs table used for prescribing
$result = $db->prepare("SELECT * FROM settings");
$result->execute();
for($i=0; $row = $result->fetch(); $i++){
$useFdaDrugsList =$row['fda_user'];


if($useFdaDrugsList==1){

// Fetch records
$result = $db->prepare("SELECT id,ActiveIngredient AS Name,DrugName as brand FROM meds WHERE ActiveIngredient LIKE :Name OR DrugName LIKE :Name LIMIT :limit");
}
else{
$result = $db->prepare("SELECT drug_id as id, generic_name as Name,brand_name AS brand  FROM drugs WHERE generic_name  LIKE :Name OR brand_name LIKE :Name LIMIT :limit");
}
$result->bindValue(':Name', $search.'%', PDO::PARAM_STR);
$result->bindValue(':limit', (int)$numberofrecords, PDO::PARAM_INT);
$result->execute();
$drugList = $result->fetchAll();

}

$data = array();

// Read Data
foreach($drugList as $drug){
$data[] = array(
"id" => $drug['id'],
"text" => $drug['Name']."--".$drug['brand']

);
}

echo json_encode($data);
exit();
}
